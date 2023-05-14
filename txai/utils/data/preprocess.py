import random
import torch
import numpy as np
import pandas as pd
import sys, os
from .utils_phy12 import *

base_path = '/home/owq978/TimeSeriesXAI/PAMdata/PAMAP2data/'

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class PAMchunk:
    '''
    Class to hold chunks of PAM data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = len(self.X)           
        idx = random.choice(np.arange(n_samp))
        
        static_idx = None if self.static is None else self.static[idx]
        print('In chunk', self.time.shape)
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

class RWDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        time = self.times[:,idx]
        y = self.y[idx]
        return x, time, y 


def process_PAM(split_no = 1, device = None, base_path = base_path, gethalf = False):
    split_path = 'splits/PAMAP2_split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle=True)

    Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)

    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    #return Ptrain, Pval, Ptest, ytrain, yval, ytest

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_tensor = Ptrain
    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    if gethalf:
        Ptrain_tensor = Ptrain_tensor[:,:,:(Ptrain_tensor.shape[-1] // 2)]
        Pval_tensor = Pval_tensor[:,:,:(Pval_tensor.shape[-1] // 2)]
        Ptest_tensor = Ptest_tensor[:,:,:(Ptest_tensor.shape[-1] // 2)]

    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    train_chunk = PAMchunk(Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = PAMchunk(Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = PAMchunk(Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def zip_x_time_y(X, time, y):
    # Break up all args into lists in first dimension:
    Xlist = [X[:,i,:].unsqueeze(1) for i in range(X.shape[1])]
    timelist = [time[:,i].unsqueeze(dim=1) for i in range(time.shape[1])]
    ylist = [y[i] for i in range(y.shape[0])]

    return list(zip(Xlist, timelist, ylist))

class ECGchunk:
    '''
    Class to hold chunks of ECG data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = self.X.shape[1]           
        idx = random.choice(np.arange(n_samp))

        static_idx = None if self.static is None else self.static[idx]
        #print('In chunk', self.time.shape)
        return self.X[idx,:,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def get_all(self):
        static_idx = None # Doesn't support non-None 
        return self.X, self.time, self.y, static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:], \
            self.time[:,idx], \
            self.y[idx].unsqueeze(dim=0)
            #static_idx

def mask_normalize_ECG(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    P_tensor = P_tensor.numpy()
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    #Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pnorm_tensor
    #return Pnorm_tensor

def tensorize_normalize_ECG(P, y, mf, stdf):
    F, T = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize_ECG(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor

ecg_base_path = '/home/owq978/TimeSeriesXAI/ECGdata/ECG'
def process_ECG(split_no = 1, device = None, base_path = ecg_base_path):

    # train = torch.load(os.path.join(loc, 'train.pt'))
    # val = torch.load(os.path.join(loc, 'val.pt'))
    # test = torch.load(os.path.join(loc, 'test.pt'))

    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    # Ptrain, Pval, Ptest = train['samples'].transpose(1, 2), val['samples'].transpose(1, 2), test['samples'].transpose(1, 2)
    # ytrain, yval, ytest = train['labels'], val['labels'], test['labels']

    X, y = torch.load(os.path.join(base_path, 'all_ECG.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    # print('X', Ptrain_tensor)
    # print('time', Ptrain_time_tensor)
    print('X', Ptrain_tensor.shape)
    print('time', Ptrain_time_tensor.shape)
    # print('time of 0', Ptrain_time_tensor.sum())
    # print('train under 0', (Ptrain_tensor > 1e-10).sum() / Ptrain_tensor.shape[1])
    #print('After s-permute', Ptrain_time_tensor.shape)
    #exit()
    train_chunk = ECGchunk(Ptrain_tensor, None, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor, None, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor, None, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

mitecg_base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG'
def process_MITECG(split_no = 1, device = None, hard_split = False, normalize = False, exclude_pac_pvc = False, balance_classes = False, div_time = False, 
        need_binarize = False, base_path = mitecg_base_path):

    split_path = 'split={}.pt'.format(split_no)
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()

        # Make times on the fly:
        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)

        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
        
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))

    Ptrain, time_train, ytrain = X[:,idx_train,:].float(), times[:,idx_train], y[idx_train].long()
    Pval, time_val, yval = X[:,idx_val,:].float(), times[:,idx_val], y[idx_val].long()
    Ptest, time_test, ytest = X[:,idx_test,:].float(), times[:,idx_test], y[idx_test].long()

    if normalize:

        # Get mean, std of the whole sample from training data, apply to val, test:
        mu = Ptrain.mean()
        std = Ptrain.std()
        Ptrain = (Ptrain - mu) / std
        Pval = (Pval - mu) / std
        Ptest = (Ptest - mu) / std

        # Normalize each sample to between 0,1:
        # samp_len = Ptrain.shape[0]
        # batch_mins = Ptrain.min(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # batch_maxes = Ptrain.max(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # Ptrain = (Ptrain -  batch_mins) / batch_maxes 

        # batch_mins = Pval.min(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # batch_maxes = Pval.max(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # Pval = (Pval -  batch_mins) / batch_maxes 

        # batch_mins = Ptest.min(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # batch_maxes = Ptest.max(dim=0)[0].unsqueeze(0).repeat(samp_len, 1, 1)
        # Ptest = (Ptest -  batch_mins) / batch_maxes 

    if div_time:
        time_train = time_train / 60.0
        time_val = time_val / 60.0
        time_test = time_test / 60.0

    if exclude_pac_pvc:
        train_mask_in = (ytrain < 3)
        Ptrain = Ptrain[:,train_mask_in,:]
        time_train = time_train[:,train_mask_in]
        ytrain = ytrain[train_mask_in]

        val_mask_in = (yval < 3)
        Pval = Pval[:,val_mask_in,:]
        time_val = time_val[:,val_mask_in]
        yval = yval[val_mask_in]

        test_mask_in = (ytest < 3)
        Ptest = Ptest[:,test_mask_in,:]
        time_test = time_test[:,test_mask_in]
        ytest = ytest[test_mask_in]
    
    if need_binarize:
        ytrain = (ytrain > 0).long()
        ytest = (ytest > 0).long()
        yval = (yval > 0).long()

    if balance_classes:
        diff_to_mask = (ytrain == 0).sum() - (ytrain == 1).sum()
        all_zeros = (ytrain == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptrain.shape[1])])
        print('Num before', (ytrain == 0).sum())
        Ptrain = Ptrain[:,to_mask_in,:]
        time_train = time_train[:,to_mask_in]
        ytrain = ytrain[to_mask_in]
        print('Num after 0', (ytrain == 0).sum())
        print('Num after 1', (ytrain == 1).sum())

        diff_to_mask = (yval == 0).sum() - (yval == 1).sum()
        all_zeros = (yval == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Pval.shape[1])])
        print('Num before', (yval == 0).sum())
        Pval = Pval[:,to_mask_in,:]
        time_val = time_val[:,to_mask_in]
        yval = yval[to_mask_in]
        print('Num after 0', (yval == 0).sum())
        print('Num after 1', (yval == 1).sum())

        diff_to_mask = (ytest == 0).sum() - (ytest == 1).sum()
        all_zeros = (ytest == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptest.shape[1])])
        print('Num before', (ytest == 0).sum())
        Ptest = Ptest[:,to_mask_in,:]
        time_test = time_test[:,to_mask_in]
        ytest = ytest[to_mask_in]
        print('Num after 0', (ytest == 0).sum())
        print('Num after 1', (ytest == 1).sum())

    train_chunk = ECGchunk(Ptrain, None, time_train, ytrain, device = device)
    val_chunk = ECGchunk(Pval, None, time_val, yval, device = device)
    test_chunk = ECGchunk(Ptest, None, time_test, ytest, device = device)

    print('Num after 0', (yval == 0).sum())
    print('Num after 1', (yval == 1).sum())
    print('Num after 0', (ytest == 0).sum())
    print('Num after 1', (ytest == 1).sum())

    if hard_split:
        gt_exps = saliency.transpose(0,1).unsqueeze(-1)[:,idx_test,:]
        if exclude_pac_pvc:
            gt_exps = gt_exps[:,test_mask_in,:]
        return train_chunk, val_chunk, test_chunk, gt_exps
    else:
        return train_chunk, val_chunk, test_chunk

class EpiDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, augment_negative = None):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

        #self.augment_negative = augment_negative
        if augment_negative is not None:
            mu, std = X.mean(dim=1), X.std(dim=1, unbiased = True)
            num = int(self.X.shape[1] * augment_negative)
            Xnull = torch.stack([mu + torch.randn_like(std) * std for _ in range(num)], dim=1).to(self.X.get_device())

            self.X = torch.cat([self.X, Xnull], dim=1)
            extra_times = torch.arange(self.X.shape[0]).to(self.X.get_device())
            self.times = torch.cat([self.times, extra_times.unsqueeze(1).repeat(1, num)], dim = -1)
            self.y = torch.cat([self.y, (torch.ones(num).to(self.X.get_device()).long() * 2)], dim = 0)

        # print('X', self.X.shape)
        # print('times', self.times.shape)
        # print('y', self.y.shape)
        # exit()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

epi_base_path = '/home/owq978/TimeSeriesXAI/ECGdata/Epilepsy'
def process_Epilepsy(split_no = 1, device = None, base_path = epi_base_path):

    # train = torch.load(os.path.join(loc, 'train.pt'))
    # val = torch.load(os.path.join(loc, 'val.pt'))
    # test = torch.load(os.path.join(loc, 'test.pt'))

    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    # Ptrain, Pval, Ptest = train['samples'].transpose(1, 2), val['samples'].transpose(1, 2), test['samples'].transpose(1, 2)
    # ytrain, yval, ytest = train['labels'], val['labels'], test['labels']

    X, y = torch.load(os.path.join(base_path, 'all_epilepsy.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    # print('X', Ptrain_tensor)
    # print('time', Ptrain_time_tensor)
    print('X', Ptrain_tensor.shape)
    print('time', Ptrain_time_tensor.shape)
    # print('time of 0', Ptrain_time_tensor.sum())
    # print('train under 0', (Ptrain_tensor > 1e-10).sum() / Ptrain_tensor.shape[1])
    #print('After s-permute', Ptrain_time_tensor.shape)
    #exit()
    train_chunk = ECGchunk(Ptrain_tensor, None, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor, None, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor, None, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def decomposition_statistics(pool_layer, X):

    # Decomposition by trend layer:
    trend = pool_layer(X)
    seasonal = X - trend

    d = {
        'mu_trend': trend.mean(dim=1),
        'std_trend': trend.std(unbiased = True, dim = 1),
        'mu_seasonal': seasonal.mean(dim=1),
        'std_seasonal': seasonal.std(unbiased = True, dim = 1)
    }

    return d

boiler_base_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Boiler"

def process_Boiler(split_no = 1, device = None, base_path = boiler_base_path, normalize = False):
    x_full = torch.load(os.path.join(base_path, 'xfull.pt')).to(device).float()
    y_full = torch.load(os.path.join(base_path, 'yfull.pt')).to(device).long()
    sfull = torch.load(os.path.join(base_path, 'sfull.pt')).to(device).float()
    print('s', sfull.shape)
    print('xfull', x_full.shape)
    print('yfull', y_full.shape)
    # exit()

    T_full = torch.zeros(36, x_full.shape[1]).to(device)
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    train_d = [x_full[:,idx_train,:], T_full[:,idx_train], y_full[idx_train]]
    val_d = [x_full[:,idx_val,:], T_full[:,idx_val], y_full[idx_val]]
    test_d = [x_full[:,idx_test,:], T_full[:,idx_test], y_full[idx_test]]

    # if normalize:
    #     # Get mean, std of the whole sample from training data, apply to val, test:
    #     mu = train_d[0].mean(dim=1).unsqueeze(1)
    #     std = train_d[0].std(dim=1).unsqueeze(1).repeat(1,train_d[0].shape[1],1)
    #     train_d[0] = (train_d[0] - mu.repeat(1,train_d[0].shape[1],1)) / std.repeat(1,train_d[0].shape[1],1)
    #     val_d[0] = (val_d[0] - mu.repeat(1,val_d[0].shape[1],1)) / std.repeat(1,val_d[0].shape[1],1)
    #     test_d[0] = (test_d[0] - mu.repeat(1,test_d[0].shape[1],1)) / std.repeat(1,test_d[0].shape[1],1)

    stest = sfull[:,idx_test,:]

    return train_d, val_d, test_d, stest


def process_Boiler_OLD(split_no = 1, device = None, base_path = boiler_base_path):
    data = pd.read_csv(os.path.join(base_path, 'full.csv')).values
    data = data[:, 2:]  #remove time step

    window_size = 6
    segments_length = [1, 2, 3, 4, 5, 6]

    # Load path

    print('positive sample size:',sum(data[:,-1]))
    feature, label = [], []
    for i in range(window_size - 1, len(data)):
        label.append(data[i, -1])

        sample = []
        for length in segments_length:
            a = data[(i- length + 1):(i + 1), :-1]
            a = np.pad(a,pad_width=((0,window_size -length),(0,0)),mode='constant')# padding to [window_size, x_dim]
            sample.append(a)

        sample = np.array(sample)
        sample = np.transpose(sample,axes=((2,0,1)))[:,:,:]

        feature.append(sample)

    feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    x_full = torch.tensor(feature.reshape(*feature.shape[:-2], -1)).permute(2,0,1)
    y_full = torch.from_numpy(label)

    # Make times:
    T_full = torch.zeros(36, x_full.shape[1])
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    # Now split:
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    x_full, T_full, y_full = x_full.to(device), T_full.to(device), y_full.to(device).long()

    train_d = (x_full[:,idx_train,:], T_full[:,idx_train], y_full[idx_train])
    val_d = (x_full[:,idx_val,:], T_full[:,idx_val], y_full[idx_val])
    test_d = (x_full[:,idx_test,:], T_full[:,idx_test], y_full[idx_test])

    return train_d, val_d, test_d