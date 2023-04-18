import timesynth as ts
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

def generate_seq(T = 500, D = 15, class_num = 0):

    '''
    class_num must be in [0,1]
    '''

    assert class_num in [0,1,2,3], 'class_num must be in [0,1,2]'

    # Sample:
    samp = np.zeros((T, D))

    for di in range(D):
        noise = ts.noise.GaussianNoise(std=0.01)
        x = ts.signals.NARMA(order=2,seed=random.seed())
        x_ts = ts.TimeSeries(x, noise_generator=noise)
        x_sample, signals, errors = x_ts.sample(np.array(range(T)))
        samp[:,di] = x_sample

    if class_num == 0: # Null class - make no modifications
        return samp, [None]

    # Make combinations:

    # Sample sequence length through random uniform
    if D > 1:
        imp_sensors = np.random.choice(np.arange(D), size = (2,), replace = False)
    else:
        imp_sensors = [0, 0]

    min_x, max_x = np.min(samp), np.max(samp)

    # Inject centered around mean of sample, linear between 5 x min, 5 x max
    bounds = (-5 * np.abs(min_x), 5 * np.abs(max_x)) # Ensures we elongate values
    rev_bounds = (bounds[1], bounds[0])

    # Choose bounds of samples based on class number:
    if class_num == 1:
        B = (rev_bounds, rev_bounds)
    elif class_num == 2:
        B = (bounds, bounds)
    elif class_num == 3:
        B = (bounds, rev_bounds)

    coords = []

    prev_seq = []

    j = 0
    for i in imp_sensors:
        b = B[j]
        j += 1
        # Iterate over important sensors
        seqlen = np.random.randint(low = 10, high = 20, size = 1)[0]
        if len(prev_seq) > 0:
            psl, pts = prev_seq[0]
            imp_time = np.random.randint(low = psl + pts, high = T - seqlen)
        else:
            imp_time = np.random.randint(low = 0, high = T // 2, size = 1)[0]
        amp = np.random.poisson(lam = 10, size = 1)[0]

        prev_seq.append((seqlen, imp_time))

        # Decide slope (y_2 - y_1) / (x_2, x_1):
        x_1, x_2 = -(seqlen // 2), ((seqlen // 2) + seqlen % 2)
        # if (x_2 - x_1) != seqlen:
        #     # Add differerence to x_2:
        #     x_2 += ((x_2 - x_1) - seqlen)
        y_1, y_2 = b[0], b[1]
        slope = (y_2 - y_1) / (x_2 - x_1)

        # Evaluate function over time:
        newseq = lambda x: slope * x + np.sin(amp * x)

        # x-values should be sequence around 0
        xvals = np.arange(x_1,x_2)
        samp[imp_time:(imp_time + seqlen),i] = newseq(xvals)

        # Pick out coordinates:
        coords += list(zip(list(range(imp_time, imp_time+seqlen)), [i] * seqlen))

    return samp, coords

def generate_spike_dataset(N = 1000, T = 500, D = 15):

    # Get even number of samples for each class:
    # 3 classes: null class, class 1, class 2
    class_count = [(N // 4)] * 3
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
            Xi, locs = generate_seq(T, D, class_num = i)
            X[total_count,:,:] = Xi
            times[total_count,:] = np.arange(1,T+1) # Steadily increasing times
            y[total_count] = i # Needs to be zero-indexed
            gt_exps.append(locs)
            total_count += 1

    return X, times, y, gt_exps
    

class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, N = 1000, T = 500, D = 15):
        
        self.X, self.times, self.y, _ = generate_spike_dataset(
            N = N, T = T, D = D
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
        try:
            for i, j in gt_exp[n]:
                Xgt[i,n,j] = 1
        except TypeError:
            continue
    return Xgt

def convert_torch(X, times, y):
    X = torch.from_numpy(X).transpose(0,1)
    times = torch.from_numpy(times).transpose(0,1)
    y = torch.from_numpy(y).long()

    return X, times, y

def get_all_spike_loaders(Ntrain = 1000, T = 500, D = 15, Nval = 100, Ntest = 300):

    config_args = (T, D)

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

def visualize_some(dataset):

    X, times, y = dataset['test']
    gt_exps = dataset['gt_exps']

    # Pick out with each label:
    uni = torch.unique(y).numpy()

    for i in uni:

        choice = np.random.choice((y == i).nonzero(as_tuple=True)[0].numpy())

        Xc, gtc = X[:,choice,:].numpy(), gt_exps[:,choice,:].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(Xc)
        ax2.imshow(gtc)
        fig.suptitle('Label = {}'.format(i))
        plt.savefig('example_{}.png'.format(i))


def print_tuple(t):
    print('X', t[0].shape)
    print('time', t[1].shape)
    print('y', t[2].shape)

if __name__ == '__main__':

    for i in range(5):
        print(f'Split {i + 1} ' + '-' * 20)
        train_dataset, val, test, gt_exps = get_all_spike_loaders(Ntrain=5000, Nval=100, Ntest=1000, T = 200, D = 1)

        dataset = {
            'train_loader': train_dataset,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        base = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets'
        torch.save(dataset, base + '/SeqCombSingleBetter/split={}.pt'.format(i + 1))
        
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        #visualize_some(dataset)