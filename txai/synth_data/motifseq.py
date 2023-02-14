import timesynth as ts
import numpy as np
import pickle as pkl
import os, math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt

import torch
from tqdm import trange, tqdm

from txai.synth_data.synth_data_base import GenerateSynth, print_tuple, visualize_some

# UTILS --------------------------------------------------------------------------
def gen_motif(max_x, min_x, c = 1, low_len = 8, high_len = 16):

    assert max_x > min_x, 'Max comes first in gen_motif!'
    # Sample length:
    length = np.random.randint(low_len,  high_len)

    if c == 1:
        # Monotonic increasing:
        slope = np.random.randint(length,  length*2)
        seq = np.linspace(min_x, max_x, num = length) * slope
        
    elif c == 2:
        # Monotonic decreasing:
        slope = np.random.randint(length,  length*2)
        seq = np.linspace(max_x, min_x, num = length) * slope

    elif c == 3:
        # Concave down (increasing then decreasing)
        seq = np.zeros(length)
        midpoint = math.ceil(length / 2)

        slope = np.random.randint(length*2,  length*4)
        seq[:midpoint] = np.linspace(min_x, max_x, num = midpoint) * slope
        seq[(midpoint-1):] = np.linspace(max_x, min_x, num = (length - midpoint + 1)) * slope
    
    else:
        # Concave up (decreasing then increasing)
        seq = np.zeros(length)
        midpoint = math.ceil(length / 2)

        slope = np.random.randint(length*2,  length*4)
        seq[:midpoint] = np.linspace(max_x, min_x, num = midpoint) * slope
        seq[(midpoint-1):] = np.linspace(min_x, max_x, num = (length - midpoint + 1)) * slope

    return seq

def get_loc_given_seq(seq, T):
    buffer = seq.shape[0] + 3
    loc = np.random.choice(np.arange(buffer, T - buffer))
    return loc, loc + seq.shape[0] 

# UTILS --------------------------------------------------------------------------

class MotifSeq(GenerateSynth):

    def __init__(self, T, D, season_thresh = 0.3):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(MotifSeq, self).__init__(T, D, 9)
        self.season_thresh = season_thresh


    def generate_seq(self, class_num):

        '''
        class_num must be in [0,1]
        '''

        assert class_num in list(range(9)), 'class_num must be in [0,1,2]'

        # Sample:
        # print('T, d', self.T, self.D)
        samp = np.zeros((self.T, self.D))

        for di in range(self.D):
            noise = ts.noise.GaussianNoise(std=0.01)
            x = ts.signals.NARMA(order=2,seed=random.seed())
            x_ts = ts.TimeSeries(x, noise_generator=noise)
            x_sample, signals, errors = x_ts.sample(np.array(range(self.T)))
            samp[:,di] = x_sample

        if class_num == 0: # Null class - make no modifications
            return samp, [None]

        # Decifer class number:
        motif_type = ((class_num - 1) % 4) + 1
        longseason = (class_num > 4)
        season_thresh_int = int(self.T * self.season_thresh)

        # Sample sequence length through random uniform
        important_sensor = np.random.choice(np.arange(self.D))

        min_x, max_x = np.min(samp), np.max(samp)

        coords = []

        j = 0
        for i in range(self.D): # Iterate over all sensors

            if i == important_sensor:

                motifs = [
                    gen_motif(max_x, min_x, motif_type), 
                    gen_motif(max_x, min_x, motif_type)
                ]
                #print('lseason', longseason)
                
                if longseason:
                    # Decide on length of season:
                    upperbound = self.T - sum([m.shape[0] for m in motifs]) - 6 # 6 acts as 3 buffer on each end
                    seasonlen = np.random.choice(np.arange(season_thresh_int + 2,  upperbound))
                    
                else:
                    # Decide on length of season:
                    seasonlen = np.random.choice(np.arange(3,  season_thresh_int - 2))

                # Decide placement of first
                upperbound_placement = self.T - seasonlen - motifs[1].shape[0] - 3
                start = np.random.choice(np.arange(motifs[0].shape[0] + 3, upperbound_placement)) 
                end = start + seasonlen
                
                # Set the motifs in the sequence:
                m1size = motifs[0].shape[0]
                lower = start - m1size
                # print('lower', lower)
                # print('start', start)
                # print('motifs', motifs[0])
                # print('samp', samp[lower:start,i])
                samp[lower:start,i] = motifs[0]

                coords1 = [(j, i) for j in np.arange(lower, start)]

                m2size = motifs[1].shape[0]
                upper = end + m2size
                samp[end:upper,i] = motifs[1]

                coords2 = [(j, i) for j in np.arange(end, upper)]

                coords = coords1 + coords2 # Add all relevant coordinates

            else:
                toplant = gen_motif(max_x, min_x, np.random.choice(np.arange(1, 5)))
                loc1, loc2 = get_loc_given_seq(toplant, T = self.T)
                samp[loc1:loc2,i] = toplant

        return samp, coords

if __name__ == '__main__':

    gen = MotifSeq(T = 100, D = 4)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=10000, Nval=500, Ntest=2000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/MotifSeq/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        print('Visualizing')
        visualize_some(dataset, save_prefix = 'motif') 