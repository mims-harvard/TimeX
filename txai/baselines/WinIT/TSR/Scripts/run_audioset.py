import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler

from .Models.TCN import TCN
from .train_models import train_model
from .interpret import run_saliency_methods
from .getSaliencyMapMetadata import getSaliencyMapMetadata


def get_data(path):
    def load_data(hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            x = hf.get('x')
            y = hf.get('y')
            video_id_list = hf.get('video_id_list')
            x = np.array(x)
            y = list(y)
            video_id_list = list(video_id_list)

        return x, y, video_id_list

    def uint8_to_float32(x):
        return (np.float32(x) - 128.) / 128.

    def bool_to_float32(y):
        return np.float32(y)

    (x, y, video_id_list) = load_data(path)
    x = uint8_to_float32(x)  # shape: (N, 10, 128)
    y = bool_to_float32(y)  # shape: (N, 527)
    return x, y


def single_class_loader_from_path(path, batch_size, classify_on, balance=False):
    x, y = get_data(path)

    # Train to classify on a single label for simplicity
    y = y[:, classify_on]

    if balance:
        _, num_timesteps, num_features = x.shape
        # Balance classes via random undersampling
        x, y = RandomUnderSampler(sampling_strategy='majority').fit_resample(x.reshape(-1, num_timesteps * num_features), y)
        x = x.reshape(-1, num_timesteps, num_features)
        print(f'After balancing, dataset size for class {classify_on} is {len(y)}')

    dataset = TensorDataset(torch.from_numpy(x).double(), torch.from_numpy(y).double())

    return DataLoader(dataset, batch_size=batch_size), len(y)


train_path = 'Audioset/packed_features/bal_train.h5'
test_path = 'Audioset/packed_features/eval.h5'
data_name = 'Audioset'

batch_size = 32

# In class or not in class
num_classes = 2
classify_on = 67

hidden_size = 5
levels = 3
kernel_size = 4
dropout = 0.1

criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 500
learning_rate = 0.001

saliency_methods = ['Grad', 'Grad_TSR', 'IG', 'IG_TSR', 'DLS', 'DLS_TSR', 'FIT']
saliency_dir = 'Audioset/Saliency_Values/'
saliency_maps_graphs_dir = 'Audioset/Saliency_Maps/'


def main():
    train_loader, train_size = single_class_loader_from_path(train_path, batch_size, classify_on, balance=True)
    test_loader, test_size = single_class_loader_from_path(test_path, batch_size, classify_on, balance=True)

    _, num_timesteps, num_features = iter(train_loader).next()[0].shape

    # Just run on TCN for now
    num_channels = [hidden_size] * (levels - 1) + [num_timesteps]
    model = TCN(num_features, num_classes, num_channels, kernel_size, dropout)

    model_path = train_model(model, 'TCN', data_name, criterion, train_loader, test_loader, device, num_timesteps,
                             num_features, num_epochs, data_name, learning_rate)

    model = torch.load(model_path, map_location=device)

    run_saliency_methods(saliency_methods, model, (test_size, num_timesteps, num_features), train_loader, test_loader,
                         device, 'TCN', data_name, saliency_dir)
    getSaliencyMapMetadata(saliency_dir, saliency_maps_graphs_dir, [0])


if __name__ == '__main__':
    main()
