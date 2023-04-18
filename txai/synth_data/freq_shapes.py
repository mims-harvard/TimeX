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

from txai.synth_data.synth_data_base import GenerateSynth, print_tuple, visualize_some, plot_visualize_some

class FreqShapes(GenerateSynth):

    class_prop_map = {
        0: (0, 10),
        1: (1, 10),
        2: (0, 17),
        3: (1, 17)
    }

    def __init__(self, T, noise = 0.25):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(FreqShapes, self).__init__(T, D=1, n_classes=4)
        self.noise = noise


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2,3], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, 1))

        shape, freq = self.class_prop_map[class_num]

        start = np.random.choice(np.arange(self.T - freq))
        spike_centers = np.arange(start, self.T, step=freq, dtype=int)
        
        bottom = (spike_centers - 1)
        top = (spike_centers + 2) # Need one more to account for exclusive top-end indexing
        valid = (bottom >= 0) & (top <= self.T)
        bottom, top = bottom[valid], top[valid] # Filter for valid indices
        
        gt_exp = []

        for b, t in zip(bottom, top):

            # Make spikes:
            if shape == 0:
                # Downward spike
                samp[b:t,0] = 0.25 * (np.arange(-1, 2)) ** 2 - 2
                    
            elif shape == 1:
                # Upward spike
                samp[b:t,0] = -0.25 * (np.arange(-1, 2)) ** 2 + 2

            gt_exp += list(range(b, t))

        samp += np.random.normal(loc = 0, scale = self.noise, size = (self.T,1))

        return samp, [(g, 0) for g in gt_exp]
    
class FreqShapesUpDown(GenerateSynth):

    class_prop_map = {
        0: (0, 10),
        1: (1, 10),
        2: (0, 17),
        3: (1, 17)
    }

    def __init__(self, T, noise = 0.25):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(FreqShapesUpDown, self).__init__(T, D=1, n_classes=4)
        self.noise = noise


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2,3], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, 1))

        shape, freq = self.class_prop_map[class_num]

        start = np.random.choice(np.arange(self.T - freq - 5))
        spike_centers = np.arange(start, self.T, step=freq, dtype=int)
        
        bottom = (spike_centers - 3)
        top = (spike_centers + 3) # Need one more to account for exclusive top-end indexing
        valid = (bottom >= 0) & (top <= self.T)
        bottom, top = bottom[valid], top[valid] # Filter for valid indices
        
        gt_exp = []

        for b, t in zip(bottom, top):
            
            downspike = 0.25 * (np.arange(-1, 2)) ** 2 - 2
            upspike = -0.25 * (np.arange(-1, 2)) ** 2 + 2

            # Make spikes:
            if shape == 0:
                # Downward then up spike
                samp[b:t,0] = np.concatenate((downspike, upspike))
                    
            elif shape == 1:
                # Upward then downward spike
                samp[b:t,0] = np.concatenate((upspike, downspike))

            gt_exp += list(range(b, t))

        samp += np.random.normal(loc = 0, scale = self.noise, size = (self.T,1))

        return samp, [(g, 0) for g in gt_exp]

if __name__ == '__main__':

    gen = FreqShapesUpDown(T = 50)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        torch.save(dataset, '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        #plot_visualize_some(dataset) 