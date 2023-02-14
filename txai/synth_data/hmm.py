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

class HMM(GenerateSynth):

    def __init__(self, T, D, n_classes):
        '''
        D: dimension of samples (# sensors)
        T: length of time for each sample
        '''
        super(HMM, self).__init__(T, D, n_classes)


    def generate_seq(self, class_num):
        '''
        class_num corresponds to the number of state transitions
        '''

        cond = True
        repeat = False

        while cond:

            # if repeat:
            #     print('Run again')

            # Construct transition matrix:
            tmat = np.eye(self.D)
            other_prob = min(class_num / self.T / self.D, 1.0 / self.D)
            for i in range(tmat.shape[0]):
                tmat[i,:i] = other_prob
                tmat[i,(i+1):] = other_prob
                tmat[i,i] -= other_prob * (tmat.shape[0] - 1) 

            # Make Gaussian distributions:
            range_locs = np.arange(-(self.D // 2), math.ceil(self.D / 2)) * 8.0
            #   8 separation roughly gives 0 probability of drawing same numbers in Gaussians
            dists = [lambda : np.random.normal(loc=n, scale=1.0, size = self.D) for n in range_locs]

            # Sample:
            samp = np.zeros((self.T, self.D))

            # Choose random state in which to start:
            state = np.random.randint(0, high = self.D)
            samp[0,:] = np.abs(np.random.normal(loc = 8, size = (self.D,)))
            samp[0,state] *= -1.0
            gt_exp_coords = []

            n_transitions = 0
            fin_transition = 0

            for t in range(1, self.T):

                if n_transitions < class_num:

                    if (t > self.T * 0.9):
                        # Essentially the same operation as tmat initialization ^^^
                        tmat = tmat + \
                            (tmat * (1 - np.eye(self.D))) - \
                            (np.eye(self.D) * other_prob * (tmat.shape[0] - 1))

                    # Draw random number
                    draw = np.random.uniform()
                    # Decide to switch - chooses earliest in cumsum draw
                    state_new = (np.cumsum(tmat[state,:]) > draw).nonzero()[0][0]

                    if state != state_new:
                        if len(gt_exp_coords) > 0: # Condition controls for back-to-back transitions
                            if gt_exp_coords[-1][0] != (t-1):
                                gt_exp_coords.append((t - 1, state))
                        else:
                            gt_exp_coords.append((t - 1, state))
                        gt_exp_coords.append((t, state_new))
                        n_transitions += 1
                        fin_transition = t

                    state = state_new # Switch to new state

                # If above does not catch, keep acting as if we're in our current state

                # Draw out feature vector:
                samp[t,:] = np.abs(np.random.normal(loc = 8, size = (self.D,)))
                samp[t,state] *= -1.0

            cond = (n_transitions != class_num) # Continues loop if we didn't get the right number
            repeat = True

        #print('Class num = {}, Num transitions = {}, Final Transition = {}'.format(class_num, n_transitions, [g[0] for g in gt_exp_coords]))

        return samp, gt_exp_coords

if __name__ == '__main__':

    gen = HMM(T = 50, D = 4, n_classes = 4)

    for i in range(5):
        train, val, test, gt_exps = gen.get_all_loaders(Ntrain=5000, Nval=100, Ntest=1000)

        dataset = {
            'train_loader': train,
            'val': val,
            'test': test,
            'gt_exps': gt_exps
        }

        #torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/States/split={}.pt'.format(i + 1))
        
        print('Split {} -------------------------------'.format(i+1))
        print('Val ' + '-'*20)
        print_tuple(val)
        print('\nTest' + '-'*20)
        print_tuple(test)

        print('GT EXP')
        print(gt_exps.shape)

        visualize_some(dataset, 'state')

        exit()