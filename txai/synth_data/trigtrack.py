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

from txai.synth_data.synth_data_base import GenerateSynth, print_tuple, visualize_some

class TrigTrack(GenerateSynth):

    class_wavelen_map = {
        0: 0.25,
        1: 0.5,
        2: 1.0,
        3: 2.0
    }

    def __init__(self, T, D, noise = None):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(TrigTrack, self).__init__(T, D, 4)

        self.important_sensor = np.random.choice([0, 1, 2, 3])
        self.noise = noise


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2,3], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, self.D))

        for i in range(self.D):
            if i == self.important_sensor:
                wave_len = self.class_wavelen_map[class_num]

            else:
                wave_len = np.random.choice(np.linspace(0.25, 2, num=50))

            # Amplitude randomly sampled:
            amp = np.random.choice(np.linspace(-5.0, 5.0, num=50))

            signal = amp * np.sin(wave_len * np.arange(self.T * 2))

            # Multiply by noise:
            if self.noise is not None:
                signal = signal + np.random.normal(loc=0.0, scale = self.noise, size = signal.shape)

            # Choose random starting/ending point for signal:
            start = np.random.choice(np.arange(self.T - 2))
            end = start + self.T

            samp[:,i] = signal[start:end]

        return samp, [(i, self.important_sensor) for i in range(self.T)]

if __name__ == '__main__':

    gen = TrigTrack(T = 50, D = 4, noise = 0.25)
    print('noise', gen.noise)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        torch.save(dataset, '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/TrigTrackNoise/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        visualize_some(dataset, save_prefix = 'trig_fold{}'.format(i)) 