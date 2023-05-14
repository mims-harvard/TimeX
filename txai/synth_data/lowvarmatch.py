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

class LowVarMatch(GenerateSynth):

    def __init__(self, T, D):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(LowVarMatch, self).__init__(T, D, 3)


    def generate_seq(self, class_num = 0):

        '''
        class_num must be in [0,1,2]
        '''

        assert class_num in [0,1,2], 'class_num must be in [0,1,2]'

        # Sample:
        samp = np.zeros((self.T, self.D))

        for di in range(self.D):
            noise = ts.noise.GaussianNoise(std=1)
            x = ts.signals.NARMA(order=2,seed=random.seed())
            x_ts = ts.TimeSeries(x, noise_generator=noise)
            x_sample, signals, errors = x_ts.sample(np.array(range(self.T)))
            samp[:,di] = x_sample

        if class_num == 0: # Null class - make no modifications
            return samp, [None]

        # Sample sequence length through random uniform
        imp_sensors = np.random.choice(np.arange(self.D), size = (2,), replace = False)

        coords = []
        prev_seq = []
        j = 0
        for i in imp_sensors:
            # Iterate over important sensors
            #print('it', j)
            j += 1
            seqlen = 20
            if len(prev_seq) > 0:
                psl, pts = prev_seq[0]
                if class_num == 1:
                    exclude = np.arange(pts - 20, pts + 20)
                    to_keep = np.ones(self.T, dtype = bool)
                    to_keep[exclude] = False
                    to_keep[(self.T - 20):self.T] = False
                    to_sample = np.nonzero(to_keep)[0]
                    imp_time = random.choice(to_sample)
                    #print('ts', to_sample)
                elif class_num == 2:
                    # Need overlapping:
                    imp_time = np.random.randint(low = pts - 10, high = pts + seqlen - 10)
            else:
                # Below is hardcoded for seqlen = 20
                imp_time = np.random.randint(low = 20, high = self.T - 40, size = 1)[0]
            prev_seq.append((seqlen, imp_time))

            # print('it', imp_time)
            # print('sl', seqlen)
            samp[imp_time:(imp_time + seqlen),i] = np.random.normal(loc=0.0, scale = 0.01, size = (seqlen,))

            # Make coordinates:

            # Pick out coordinates:
            if class_num == 1:
                coords += list(zip(list(range(imp_time, imp_time+seqlen)), [i] * seqlen))

        if class_num == 2:
            # Only highlight overlap:
            sl1, ts1 = prev_seq[0]
            s1 = np.arange(ts1, ts1 + sl1)
            sl2, ts2 = prev_seq[1]
            s2 = np.arange(ts2, ts2 + sl2)

            b1 = np.zeros(self.T, dtype = bool)
            b1[s1] = True
            b2 = np.zeros(self.T, dtype = bool)
            b2[s2] = True

            together = b1 & b2

            is1 = imp_sensors[0]
            coords = [(i,is1) for i in np.nonzero(together)]

            is2 = imp_sensors[1]
            coords += [(i,is2) for i in np.nonzero(together)]

        return samp, coords

if __name__ == '__main__':

    gen = LowVarMatch(T = 200, D = 2)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        plot_vis_mv(dataset)
        plt.savefig(f'example_split={i}.png')
        #exit()

        torch.save(dataset, '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/LowVarMatch/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        visualize_some(dataset, save_prefix = 'red_spike') 