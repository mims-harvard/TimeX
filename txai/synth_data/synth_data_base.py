#import timesynth as ts
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

from txai.vis.vis_saliency import vis_one_saliency

class SynthTrainDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

def print_tuple(t):
    print('X', t[0].shape)
    print('time', t[1].shape)
    print('y', t[2].shape)

def visualize_some(dataset, save_prefix = ''):

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
        plt.savefig(save_prefix + 'example_{}.png'.format(i))

def plot_vis_mv(dataset):
    X, times, y = dataset['test']
    gt_exps = dataset['gt_exps']

    uni = torch.unique(y).numpy()

    fig, ax = plt.subplots(X.shape[-1], len(uni), dpi = 200, figsize = (10, 10))

    for j, i in enumerate(uni):

        choice = np.random.choice((y == i).nonzero(as_tuple=True)[0].numpy())

        Xc, gtc = X[:,choice,:], gt_exps[:,choice,:]

        vis_one_saliency(Xc, gtc, ax, fig, col_num = j)

def plot_visualize_some(dataset):
    X, times, y = dataset['test']
    gt_exps = dataset['gt_exps']

    # Pick out with each label:
    uni = torch.unique(y).numpy()

    for i in uni:

        choice = np.random.choice((y == i).nonzero(as_tuple=True)[0].numpy())

        Xc, gtc = X[:,choice,:].numpy(), gt_exps[:,choice,:].numpy()

        plt.plot(times[:,choice], Xc[:,0])
        plt.plot(times[:,choice], gtc)
        plt.title('Label = {}'.format(i))
        #plt.savefig(save_prefix + 'example_{}.png'.format(i))
        plt.show()

class GenerateSynth:

    def __init__(self, T, D, n_classes):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        self.T, self.D, self.n_classes = T, D, n_classes

    def generate_seq(self, class_num):
        raise NotImplementedError('Must implement generate seq')

    def generate_dataset(self, N = 1000):

        # Get even number of samples for each class:
        # 3 classes: null class, class 1, class 2
        class_count = [(N // self.n_classes)] * (self.n_classes - 1)
        class_count.append(N - sum(class_count))

        gt_exps = []
        X = np.zeros((N, self.T, self.D))
        times = np.zeros((N, self.T))
        y = np.zeros(N)
        total_count = 0

        for i, n in enumerate(class_count):
            for _ in range(n):
                # n_spikes increases with count (add 1 because zero index)
                Xi, locs = self.generate_seq(class_num = i)
                X[total_count,:,:] = Xi
                times[total_count,:] = np.arange(1,self.T+1) # Steadily increasing times
                y[total_count] = i # Needs to be zero-indexed
                gt_exps.append(locs)
                total_count += 1

        return X, times, y, gt_exps

    def get_all_loaders(self, Ntrain = 1000, Nval = 100, Ntest = 300):

        Xtrain, timetrain, ytrain, _ = self.generate_dataset(Ntrain)
        Xtrain, timetrain, ytrain = self.convert_torch(Xtrain, timetrain, ytrain)
        train_dataset = SynthTrainDataset(Xtrain, timetrain, ytrain)
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        print('Train loaded')

        # Get validation tuple:
        Xval, timeval, yval, _ = self.generate_dataset(Nval)
        val_tuple = self.convert_torch(Xval, timeval, yval)
        print('Val loaded')

        # Get testing tuple:
        Xtest, timetest, ytest, gt_exps = self.generate_dataset(Ntest)
        test_tuple = self.convert_torch(Xtest, timetest, ytest)
        print('Test loaded')

        print_tuple(test_tuple)

        return train_dataset, val_tuple, test_tuple, self.apply_gt_exp_to_matrix(test_tuple[0], gt_exps)

    @staticmethod
    def apply_gt_exp_to_matrix(X, gt_exp):
        Xgt = torch.zeros_like(X)
        for n in range(len(gt_exp)):
            try:
                for i, j in gt_exp[n]:
                    Xgt[i,n,j] = 1
            except TypeError:
                continue
        return Xgt

    @staticmethod
    def convert_torch(X, times, y):
        X = torch.from_numpy(X).transpose(0,1)
        times = torch.from_numpy(times).transpose(0,1)
        y = torch.from_numpy(y).long()

        return X, times, y