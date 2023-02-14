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

class RedundantSpike(GenerateSynth):

    def __init__(self, T, D, number_spikes = 4):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(RedundantSpike, self).__init__(T, D, 3)
        self.number_spikes = number_spikes


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, self.D))

        for di in range(self.D):
            noise = ts.noise.GaussianNoise(std=0.001)
            x = ts.signals.NARMA(order=2,seed=random.seed())
            x_ts = ts.TimeSeries(x, noise_generator=noise)
            x_sample, signals, errors = x_ts.sample(np.array(range(self.T)))
            samp[:,di] = x_sample

        if class_num == 0: # Null class - make no modifications
            return samp, [None]

        prev_imp_sensors = []
        prev_imp_time = []
        num_spikes = np.random.binomial(n=self.number_spikes*2, p=0.5)
        num_spikes = max(num_spikes, 1)
        coords = []
        fill_choice = np.max(np.abs(samp))
        for _ in range(num_spikes):

            imp_sensor = np.random.choice(np.arange(self.D))
            imp_time = np.random.choice(np.arange(self.T))

            if class_num == 1:
                samp[imp_time, imp_sensor] = fill_choice * -5.0
            elif class_num == 2:
                samp[imp_time, imp_sensor] = fill_choice * 5.0

            coords.append((imp_time, imp_sensor))

        return samp, coords

if __name__ == '__main__':

    gen = RedundantSpike(T = 50, D = 4)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/RedundantSpike/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        visualize_some(dataset, save_prefix = 'red_spike') 