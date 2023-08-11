from time import time
from argparse import ArgumentParser
from pathlib import Path


import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from txai.utils.data import process_Synth
from txai.utils.data.preprocess import process_MITECG
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_Epilepsy, process_PAM
from txai.baselines.WinIT.winit.explainer.winitexplainers import WinITExplainer
from txai.baselines.WinIT.winit.utils import aggregate_scores

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WinITWrapper(WinITExplainer):
    def _model_predict(self, x, times):
        """
        Args:
            x:
                The input Tensor of shape (batch_size, num_features, num_times)
            times:
                The times Tensor of shape (batch_size, num_times)
        NOTE: rewrite with addition of times
        """

        # permute back to (time, batch, <features>)
        p = self.base_model(x.permute(2, 0, 1), times.permute(1, 0))

        # p should be of shape (batch, n_classes)
        if p.shape[-1] == 1:
            # Create a 'probability distribution' (p, 1 - p)
            prob_distribution = torch.cat((p, 1 - p), dim=1)
            return prob_distribution
        return p

    def attribute(self, x, times):
        """
        NOTE: added times for positional embedding
        Compute the WinIT attribution
        Args:
            x:
                The input Tensor of shape (batch_size, num_features, num_times)
            times:
                The times Tensor of shape (batch_size, num_times)

        Returns:
            The attribution Tensor of shape (batch_size, num_features, num_times, window_size)
            The (i, j, k, l)-entry is the importance of observation (i, j, k - window_size + l + 1)
            to the prediction at time k

        """
        self.base_model.eval()
        self.base_model.zero_grad()

        with torch.no_grad():
            tic = time()
              
            batch_size, num_features, num_timesteps = x.shape
            scores = []

            for t in tqdm(range(num_timesteps)):
                window_size = min(t, self.window_size)

                if t == 0:
                    scores.append(np.zeros((batch_size, num_features, self.window_size)))
                    continue

                # x = (num_sample, num_feature, n_timesteps)
                # times = (num_sample, n_timesteps)
                p_y = self._model_predict(x[:, :, : t + 1], times[:, : t + 1])

                iS_array = np.zeros((num_features, window_size, batch_size), dtype=float)
                for n in range(window_size):
                    time_past = t - n
                    time_forward = n + 1
                    counterfactuals = self._generate_counterfactuals(
                        time_forward, x[:, :, :time_past], x[:, :, time_past : t + 1]
                    )
                    # counterfactual shape = (num_feat, num_samples, batch_size, time_forward)
                    for f in range(num_features):
                        # repeat input for num samples
                        x_hat_in = (
                            x[:, :, : t + 1].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                        )  # (ns, bs, f, time)
                        # replace unknown with counterfactuals
                        x_hat_in[:, :, f, time_past : t + 1] = counterfactuals[f, :, :, :]

                        # Compute Q = p(y_t | tilde(X)^S_{t-n:t})
                        p_y_hat = self._model_predict(
                            x_hat_in.reshape(self.num_samples * batch_size, num_features, t + 1), 
                            times[:, : t+1].repeat([self.num_samples, 1])
                        )

                        # Compute P = p(y_t | X_{1:t})
                        p_y_exp = (
                            p_y.unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                            .reshape(self.num_samples * batch_size, p_y.shape[-1])
                        )
                        iSab_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        iSab = torch.mean(iSab_sample, dim=0).detach().cpu().numpy()
                        # For KL, the metric can be unbounded. We clip it for numerical stability.
                        iSab = np.clip(iSab, -1e6, 1e6)
                        iS_array[f, n, :] = iSab

                # Compute the I(S) array
                b = iS_array[:, 1:, :] - iS_array[:, :-1, :]
                iS_array[:, 1:, :] = b

                score = iS_array[:, ::-1, :].transpose(2, 0, 1)  # (bs, nfeat, time)

                # Pad the scores when time forward is less than window size.
                if score.shape[2] < self.window_size:
                    score = np.pad(score, ((0, 0), (0, 0), (self.window_size - score.shape[2], 0)))
                scores.append(score)
            print(f"Batch done: Time elapsed: {(time() - tic):.4f}")

            scores = np.stack(scores).transpose((1, 2, 0, 3))  # (bs, fts, ts, window_size)
            return scores


def train_generator(args):
    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShape')
    elif Dname == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingle')
    elif Dname == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingleBetter')
    elif Dname == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShapeUD')
    elif Dname == 'seqcomb_mv':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombMV')
    elif Dname == 'mitecg_hard':
        D = process_MITECG(split_no = args.split_no, device = device, hard_split = True, need_binarize = True, exclude_pac_pvc = True, base_path = Path(args.data_path) / 'MITECG-Hard')
    elif Dname == 'lowvardetect':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'LowVarDetect')
    elif Dname == 'epilepsy':
        trainEpi, val, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
    elif Dname == 'pam':
        trainEpi, val, test = process_PAM(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/PAMAP2data/', gethalf = True)

    winit_path = Path(args.models_path) / f"winit_split={args.split_no}/"


    if Dname == "mitecg_hard":
        # make ecg data the same format as everything else
        train_loader, val, test, _ = D
        train_loader = [(train_loader.X[:, i], train_loader.time[:, i], train_loader.y[i]) for i in range(train_loader.X.shape[1])]
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)
    elif Dname in {'epilepsy', 'pam'}:
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)
        #trainX = trainEpi.X
        #train_loader, val, test, _ = D
        train_loader = [(trainEpi.X[:, i], trainEpi.time[:, i], trainEpi.y[i]) for i in range(trainEpi.X.shape[1])]
    else:
        train_loader, val, test = D["train_loader"], D["val"], D["test"]

    winit = WinITExplainer(
        device, 
        num_features=test[0].shape[-1], 
        data_name=Dname, 
        path=winit_path
    )

    # NOTE: WinIT code expects time series of shape [n, features, time]
    train_input = torch.stack([train_loader[i][0].permute(1, 0) for i in range(len(train_loader))])
    train_label = torch.stack([train_loader[i][2] for i in range(len(train_loader))])
    train_ds = TensorDataset(train_input, train_label)
    # time, n, features -> n, features, time
    val_ds = TensorDataset(val[0].permute(1, 2, 0), val[2])
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)
    print("training generators...")
    start_time = time()
    results = winit.train_generators(train_loader=train_dl, valid_loader=val_dl, num_epochs=args.epochs)
    end_time = time()
    print('Time', end_time - start_time)

    plt.plot(results.train_loss_trends[0], label="train_loss")
    plt.plot(results.valid_loss_trends[0], label="valid_loss")
    plt.axvline(results.best_epochs[0], label="best_epoch", ls="--", color="black")
    plt.legend()
    plt.savefig(winit_path / "loss.png")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--models_path', type = str, help = 'path to store models')
    parser.add_argument('--data_path', default="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/", type = str, help = 'path to datasets root')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    for split_no in range(1, 6):
        args.split_no = split_no
        train_generator(args)
        exit() # TEMP REMOVE LATER