import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os, math, random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt

import torch
from tqdm import trange, tqdm

from txai.synth_data.synth_data_base import GenerateSynth, print_tuple, visualize_some, plot_vis_mv

class LowVarDetect(GenerateSynth):

    def __init__(self, T, D):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(LowVarDetect, self).__init__(T, D, 4)


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2,3], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, self.D))

        for di in range(self.D):
            noise = ts.noise.GaussianNoise(std=1)
            x = ts.signals.NARMA(order=2,seed=random.seed())
            x_ts = ts.TimeSeries(x, noise_generator=noise)
            x_sample, signals, errors = x_ts.sample(np.array(range(self.T)))
            samp[:,di] = x_sample

        # if class_num == 0: # Null class - make no modifications
        #     return samp, [None]

        # Sample sequence length through random uniform
        #imp_sensors = np.random.choice(np.arange(self.D), size = (2,), replace = False)

        i = 0 if class_num in [0,1] else 1

        seqlen = np.random.randint(low = 10, high = 20, size = 1)[0]
        imp_time = np.random.randint(low = 20, high = self.T - 40, size = 1)[0]
        loc = -1.5 if class_num in [0,2] else 1.5
        samp[imp_time:(imp_time + seqlen),i] = np.random.normal(loc=loc, scale = 0.1, size = (seqlen,))

        # Make coordinates:

        # Pick out coordinates:
        coords = list(zip(list(range(imp_time, imp_time+seqlen)), [i] * seqlen))

        return samp, coords

if __name__ == '__main__':

    gen = LowVarDetect(T = 200, D = 2)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        plot_vis_mv(dataset)
        plt.savefig(f'lvd_example_split={i}.png')
        #exit()

        torch.save(dataset, '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/LowVarDetect/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        visualize_some(dataset, save_prefix = 'red_spike') 