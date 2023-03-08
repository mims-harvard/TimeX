#import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt

import torch
from tqdm import trange, tqdm

# def generate_spikes(T = 500, D = 15, n_spikes = 4, important_sensors = 2):

#     samp = np.zeros((T, D))
#     # Choose important sensors:
#     imp_sensors = np.random.choice(np.arange(D), size = (important_sensors,), replace = False)
#     #assert (n_spikes > important_sensors), 'n_spikes must be larger than important_sensors'

#     important_sensors = min(n_spikes, important_sensors)
    
#     spikes_left = n_spikes
#     count_imp_sensors = 0

#     spike_locations = []

#     for di in range(D):
#         noise = ts.noise.GaussianNoise(std=0.001)
#         x = ts.signals.NARMA(order=2,seed=random.seed())
#         x_ts = ts.TimeSeries(x, noise_generator=noise)
#         x_sample, signals, errors = x_ts.sample(np.array(range(T)))

#         if di in imp_sensors:
#             max_samp = max(x_sample)

#             if di == max(imp_sensors):
#                 spikes_to_take = spikes_left
#             else:
#                 spikes_to_take = max(np.random.randint(spikes_left),1)
#                 spikes_left -= spikes_to_take
        
#             # Choose random indices, not overlapping:
#             spike_inds = np.random.choice(np.arange(T), 
#                 size = (spikes_to_take,), replace = False)

#             for s in spike_inds:
#                 x_sample[s] = max_samp * 3
#                 spike_locations.append((s, di))
#                 #x_sample[s-2:s+3] = np.array([3, 4, 5, 4, 3]) * max_samp

#             count_imp_sensors += 1

#         samp[:,di] = x_sample

#     return samp, spike_locations

def generate_spikes(T = 500, D = 15, n_spikes = 4, important_sensors = 2):

    samp = np.zeros((T, D))
    # Choose important sensors:
    imp_sensors = np.random.choice(np.arange(D), size = (important_sensors,), replace = False)
    #assert (n_spikes > important_sensors), 'n_spikes must be larger than important_sensors'

    #important_sensors = min(n_spikes, important_sensors)
    

    spike_locations = []
    time_pts = np.random.choice(np.arange(T), size=(n_spikes,), replace = False)

    for di in range(D):
        noise = ts.noise.GaussianNoise(std=0.001)
        x = ts.signals.NARMA(order=2,seed=random.seed())
        x_ts = ts.TimeSeries(x, noise_generator=noise)
        x_sample, signals, errors = x_ts.sample(np.array(range(T)))

        if di in imp_sensors:
            max_samp = max(x_sample)

            x_sample[time_pts] = max_samp * 3

            for t in time_pts:
                spike_locations.append((t, di))

        samp[:,di] = x_sample

    return samp, spike_locations

def generate_spike_dataset(N = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2):

    # Get even number of samples for each class:
    class_count = [(N // n_classes)] * (n_classes - 1)
    class_count.append(N - sum(class_count))

    #print('Class count', class_count)

    gt_exps = []
    X = np.zeros((N, T, D))
    times = np.zeros((N,T))
    y = np.zeros(N)
    total_count = 0

    for i, n in enumerate(class_count):
        for _ in range(n):
            # n_spikes increases with count (add 1 because zero index)
            Xi, locs = generate_spikes(T, D, n_spikes = (i + 1), important_sensors = important_sensors)
            X[total_count,:,:] = Xi
            times[total_count,:] = np.arange(1,T+1) # Steadily increasing times
            y[total_count] = i # Needs to be zero-indexed
            gt_exps.append(locs)
            total_count += 1

    return X, times, y, gt_exps
    

class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, N = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2):
        
        self.X, self.times, self.y, _ = generate_spike_dataset(
            N = N, T = T, D = D, n_classes = n_classes, important_sensors = important_sensors
        )

        self.X = torch.from_numpy(self.X).transpose(0,1)
        self.times = torch.from_numpy(self.times).transpose(0,1)
        self.y = torch.from_numpy(self.y).long()

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

def apply_gt_exp_to_matrix(X, gt_exp):
    Xgt = torch.zeros_like(X)
    for n in range(len(gt_exp)):
        for i, j in gt_exp[n]:
            Xgt[i,n,j] = 1
    return Xgt

def convert_torch(X, times, y):
    X = torch.from_numpy(X).transpose(0,1)
    times = torch.from_numpy(times).transpose(0,1)
    y = torch.from_numpy(y).long()

    return X, times, y

def get_all_spike_loaders(Ntrain = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2,
        Nval = 100, Ntest = 300):

    config_args = (T, D, n_classes, important_sensors)

    train_dataset = SpikeTrainDataset(Ntrain, *config_args)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    print('Train loaded')

    # Get validation tuple:
    Xval, timeval, yval, _ = generate_spike_dataset(Nval, *config_args)
    val_tuple = convert_torch(Xval, timeval, yval)
    print('Val loaded')

    # Get testing tuple:
    Xtest, timetest, ytest, gt_exps = generate_spike_dataset(Ntest, *config_args)
    test_tuple = convert_torch(Xtest, timetest, ytest)
    print('Test loaded')

    print_tuple(test_tuple)

    return train_dataset, val_tuple, test_tuple, apply_gt_exp_to_matrix(test_tuple[0], gt_exps)

def print_tuple(t):
    print('X', t[0].shape)
    print('time', t[1].shape)
    print('y', t[2].shape)

if __name__ == '__main__':
    X, _ = generate_spikes(50, 5, 6, 3)

    print('X', X)
    plt.imshow(X)
    plt.savefig('test.png')

    # for i in range(5):
    #     print(f'Split {i + 1} ' + '-' * 20)
    #     train_dataset, val, test, gt_exps = get_all_spike_loaders(Ntrain=5000, Nval=100, Ntest=1000, T = 200, D = 8, 
    #         n_classes = 2, important_sensors = 4)

    #     dataset = {
    #         'train_loader': train_dataset,
    #         'val': val,
    #         'test': test,
    #         'gt_exps': gt_exps
    #     }

    #     torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/Spike/spike_split={}.pt'.format(i + 1))
        
    #     print('Val ' + '-'*20)
    #     print_tuple(val)
    #     print('\nTest' + '-'*20)
    #     print_tuple(test)

    #     print('GT EXP')
    #     print(gt_exps.shape)