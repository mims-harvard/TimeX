import pathlib
import os

import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from inverse_fit import inverse_fit_attribute


def generate_sliding_window_data(inputs, labels, window_size, buffer_size, prediction_window_size):
    batch_size, num_fts, num_ts = inputs.shape
    assert num_ts >= window_size + buffer_size + prediction_window_size
    windows = []
    window_labels = []
    for t in range(num_ts - window_size - buffer_size - prediction_window_size + 1):
        windows.append(inputs[:, :, t:t + window_size])
        # TODO: Figure out what we want here
        if len(labels.shape) > 1:
            new_label = np.max(labels[:, t + window_size + buffer_size:t + window_size + buffer_size + prediction_window_size], axis=1)
        else:
            new_label = labels[:]
        window_labels.append(new_label)
    return np.concatenate(windows), np.concatenate(window_labels)


def get_model(X_train, y_train, X_test, y_test, filename=None, train=True):
    _, num_fts, num_ts = X_train.shape

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(objective='binary:logistic')

    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    if train or filename is None:
        model.fit(
            X_train,
            y_train,
            eval_metric="rmse",
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True,
            early_stopping_rounds=10)
        if filename is not None:
            model.save_model(filename)
    else:
        assert filename is not None
        model.load_model(filename)

    test_predictions = model.predict(X_test)
    AUC = metrics.roc_auc_score(y_test.flatten(), test_predictions.flatten())
    AUPR = metrics.average_precision_score(y_test.flatten(), test_predictions.flatten())
    print(f'XGB Model AUC: {AUC}, AUPR: {AUPR}')

    np.set_printoptions(precision=3)
    print("XGB feature importance matrix:")
    print(get_importance_matrix(model, num_fts, num_ts))

    return model


def get_importance_matrix(model, num_fts, window_size, importance_type="gain"):
    # Get a matrix of feature importances
    ft_imp_dict = model.get_booster().get_score(importance_type=importance_type)
    ft_imp_matrix = np.zeros((num_fts, window_size))
    for i in range(num_fts):
        for j in range(window_size):
            key = f"f{i * window_size + j}"
            if key in ft_imp_dict.keys():
                ft_imp_matrix[i, j] = ft_imp_dict[f"f{i * window_size + j}"]
            else:
                ft_imp_matrix[i, j] = 0
    return ft_imp_matrix / np.max(ft_imp_matrix)


def simple_experiment_data(shape, imp_ft, noise_mean=0, signal_mean=10):
    batch_size, num_fts, num_ts = shape
    inputs = np.random.normal(noise_mean, 1, shape)
    labels = np.zeros((batch_size, num_ts))

    for i in range(batch_size):
        transition_ts = np.random.randint(0, num_ts)
        inputs[i, imp_ft, transition_ts:] += signal_mean
        labels[i, transition_ts:] = 1

    return inputs, labels


def loader_to_np(loader):
    X_batches = []
    y_batches = []
    for X, y in loader:
        X_batches.append(X)
        y_batches.append(y)
    return np.concatenate(X_batches), np.concatenate(y_batches)


class XGBPytorchStub():
    def __init__(self, train_loader, test_loader, window_size, buffer_size, prediction_window_size, filename=None, train=True):
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.prediction_window_size = prediction_window_size

        # Used for overloading, these should not be none if the constructor is used directly
        if train_loader is not None and test_loader is not None:
            self.num_fts = next(iter(train_loader))[0].shape[1]
            self.X_train, self.y_train = generate_sliding_window_data(*loader_to_np(train_loader), self.window_size, self.buffer_size, self.prediction_window_size)
            self.X_test, self.y_test = generate_sliding_window_data(*loader_to_np(test_loader), self.window_size, self.buffer_size, self.prediction_window_size)

            self.model = get_model(self.X_train, self.y_train, self.X_test, self.y_test, filename, train)

    @classmethod
    def from_sklearn(cls, model, num_fts, window_size, buffer_size, prediction_window_size):
        pt_wrapper = cls(None, None, window_size, buffer_size, prediction_window_size)
        pt_wrapper.model = model
        pt_wrapper.num_fts = num_fts
        return pt_wrapper

    @property
    def shap_explainer(self):
        # TODO: Figure out shap's import problem and reenable shap
        return shap.Explainer(self.model)

    @property
    def imp_matrix(self):
        return get_importance_matrix(self.model, self.num_fts, self.window_size)

    def __call__(self, inputs):
        # Best we can do is run the model on the last window of input, if the input is long enough
        if inputs.shape[2] >= self.window_size:
            window = inputs[:, :, -self.window_size:].cpu().detach().numpy().reshape(inputs.shape[0], -1)
            prob_class_1 = torch.from_numpy(self.model.predict(window))
            prediction = torch.zeros((inputs.shape[0], 2))
            prediction[:, 0] = 1 - prob_class_1
            prediction[:, 1] = prob_class_1
            return prediction
        else:
            return torch.zeros(inputs.shape[0], 2)

    def eval(self):
        return self

    def to(self, device):
        return self

    def train(self):
        pass

    def timeshap(self, inputs):
        batch_size, num_fts, num_ts = inputs.shape
        inputs = inputs.cpu().detach().numpy()
        score = np.zeros(inputs.shape)
        for t in range(self.window_size, max(self.window_size, num_ts)):
            window = inputs[:, :, t - self.window_size:t].reshape(batch_size, -1)
            imp = self.shap_explainer(window).values
            imp = imp.reshape(batch_size, num_fts, self.window_size)
            score[:, :, t] = imp[:, :, -1]
        return score


def remove_and_retrain(train_loader, test_loader, window_size, buffer_size, prediction_window_size, graph_path, saliency_name, saliency_map=None):
    X_train, y_train = loader_to_np(train_loader)
    X_test, y_test = loader_to_np(test_loader)

    _, num_fts, num_ts = X_train.shape

    assert saliency_name in ["gain", "cover", "random", "greedy"] or saliency_map is not None

    auc = []

    for i in range(num_fts):
        train_windows, train_labels = generate_sliding_window_data(X_train, y_train, window_size, buffer_size, prediction_window_size)
        test_windows, test_labels = generate_sliding_window_data(X_test, y_test, window_size, buffer_size, prediction_window_size)

        cur_model = get_model(train_windows, train_labels, test_windows, test_labels)
        auc.append(metrics.roc_auc_score(test_labels.flatten(), cur_model.predict(test_windows.reshape(test_windows.shape[0], -1)).flatten()))

        if saliency_name == "greedy":
            if X_train.shape[1] > 1:
                new_auc = []
                for j in range(num_fts - i):
                    new_X_train = np.delete(X_train, j, axis=1)
                    new_X_test = np.delete(X_test, j, axis=1)
                    train_windows, train_labels = generate_sliding_window_data(new_X_train, y_train, window_size, buffer_size,
                                                                               prediction_window_size)
                    test_windows, test_labels = generate_sliding_window_data(new_X_test, y_test, window_size, buffer_size,
                                                                             prediction_window_size)
                    cur_model = get_model(train_windows, train_labels, test_windows, test_labels)
                    new_auc.append(metrics.roc_auc_score(test_labels.flatten(), cur_model.predict(
                        test_windows.reshape(test_windows.shape[0], -1)).flatten()))
                imp_fts = auc[-1] - np.array(new_auc)
            else:
                imp_fts = np.array([0])
        elif i == 0:
            if saliency_name in ["gain", "cover"]:
                imp_matrix = get_importance_matrix(cur_model, num_fts, window_size, importance_type=saliency_name)
                imp_fts = np.mean(imp_matrix, axis=-1)
            elif saliency_name == "random":
                imp_fts = np.random.random(num_fts)
            else:
                imp_fts = np.mean(saliency_map, axis=(0, 2))

        most_imp_ft = np.argmax(imp_fts)
        imp_fts = np.delete(imp_fts, most_imp_ft)
        X_train = np.delete(X_train, most_imp_ft, axis=1)
        X_test = np.delete(X_test, most_imp_ft, axis=1)

    plt.clf()
    plt.plot(list(range(num_fts)), auc)
    plt.xlim(0, num_fts)
    plt.xlabel(f"Top features removed (based on {saliency_name})")
    plt.ylim(0.5, 1)
    plt.ylabel(f"XGB AUC (AUAUC ~= {np.trapz(auc, list(range(num_fts))):.3f})")
    plt.savefig(graph_path)
    print(f'AUC ROAR graph saved to {graph_path}')


def main():
    train_size = 1000
    test_size = 100

    num_ts = 100
    num_fts = 3
    imp_ft = 1
    window_size = 5
    buffer_size = 0
    prediction_window_size = 1

    np.random.seed(0)
    X_train, y_train = simple_experiment_data((train_size, num_fts, num_ts), imp_ft)
    X_test, y_test = simple_experiment_data((test_size, num_fts, num_ts), imp_ft)

    X_train, y_train = generate_sliding_window_data(X_train, y_train, window_size, buffer_size, prediction_window_size)
    X_test, y_test = generate_sliding_window_data(X_test, y_test, window_size, buffer_size, prediction_window_size)

    model = get_model(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
