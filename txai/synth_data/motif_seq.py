import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os, math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt

import torch
from tqdm import trange, tqdm

# UTILS --------------------------------------------------------------------------
def gen_motif(max_x, min_x, c = 1, low_len = 8, high_len = 16):
    # Sample length:
    length = np.random.randint(low = low_len, high = high_len)

    if c == 1:
        # Monotonic increasing:
        slope = np.random.randint(low=length, high = length*2)
        seq = np.linspace(min_x, max_x, num = length) * slope
        
    elif c == 2:
        # Monotonic decreasing:
        slope = np.random.randint(low=length, high = length*2)
        seq = np.linspace(max_x, min_x, num = length) * slope

    elif c == 3:
        # Concave down (increasing then decreasing)
        seq = np.zeros(length)
        midpoint = math.ceil(length / 2)

        slope = np.random.randint(low=length*2, high = length*4)
        seq[:midpoint] = np.linspace(min_x, max_x, num = midpoint) * slope
        seq[(midpoint-1):] = np.linspace(max_x, min_x, num = (length - midpoint + 1)) * slope
    
    else:
        # Concave up (decreasing then increasing)
        seq = np.zeros(length)
        midpoint = math.ceil(length / 2)

        slope = np.random.randint(low=length*2, high = length*4)
        seq[:midpoint] = np.linspace(max_x, min_x, num = midpoint) * slope
        seq[(midpoint-1):] = np.linspace(min_x, max_x, num = (length - midpoint + 1)) * slope

    return seq

def get_loc_given_seq(seq, T):
    buffer = seq.shape[0] + 3
    loc = np.random.choice(np.arange(buffer, T - buffer))
    return loc, loc + seq.shape[0] 

# UTILS --------------------------------------------------------------------------

def generate_seq(T = 500, D = 15, class_num = 0, season_thresh = 0.3):

    '''
    class_num must be in [0,1]
    '''

    assert class_num in list(range(9)), 'class_num must be in [0,1,2]'

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

    # Decifer class number:
    motif_type = (class_num - 1) % 4
    longseason = (class_num > 4)
    season_thresh_int = int(T * season_thresh)

    # Sample sequence length through random uniform
    important_sensor = np.random.choice(np.arange(D))

    min_x, max_x = np.min(samp), np.max(samp)

    coords = []

    j = 0
    for i in range(D): # Iterate over all sensors

        if i == important_sensor:

            motifs = [
                gen_motif(max_x, min_x, motif_type), 
                gen_motif(max_x, min_x, motif_type)
            ]
            
            if longseason:
                # Decide on length of season:
                upperbound = T - sum([m.shape[0] for m in motifs]) - 6 # 6 acts as 3 buffer on each end
                seasonlen = np.random.choice(np.arange(low = season_thresh_int + 2, high = upperbound))
                
            else:
                # Decide on length of season:
                seasonlen = np.random.choice(np.arange(low = 3, high = season_thresh_int - 2))

            # Decide placement of first
            upperbound_placement = T - seasonlen - 3
            start = np.random.choice(np.arange(3, upperbound_placement)) 
            end = start + seasonlen
            
            # Set the motifs in the sequence:
            m1size = motifs[0].shape[0]
            lower = start - m1size
            samp[i,lower:start] = motifs[0]

            coords1 = [(i, j) for j in np.arange(lower, start)]

            m2size = motifs[1].shape[0]
            upper = end + m2size
            samp[i,end:upper] = motifs[1]

            coords2 = [(i, j) for j in np.arange(end, upper)]

            coords = coords1 + coords2 # Add all relevant coordinates

        else:
            toplant = gen_motif(np.random.choice(np.arange(1, 5)))
            loc1, loc2 = get_loc_given_seq(toplant, T = T)
            samp[loc1:loc2,i] = toplant

    return samp, coords

def generate_spike_dataset(N = 1000, T = 500, D = 15):

    # Get even number of samples for each class:
    # 3 classes: null class, class 1, class 2
    class_count = [(N // 8)] * 5
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
    

class MotifSeqTrainDataset(torch.utils.data.Dataset):
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
    # X, _ = generate_spikes(50, 5, 1)

    # print('X', X)
    # plt.imshow(X)
    # plt.savefig('test.png')

    for i in range(5):
        print(f'Split {i + 1} ' + '-' * 20)
        train_dataset, val, test, gt_exps = get_all_spike_loaders(Ntrain=5000, Nval=100, Ntest=1000, T = 50, D = 4)

        dataset = {
            'train_loader': train_dataset,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/SeqComb/scomb_split={}.pt'.format(i + 1))
        
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        #visualize_some(dataset)