import argparse
import torch
import time
import numpy as np

import sys
sys.path.append('..')
sys.path.append('../..')

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.trainers.train_transformer import train
from txai.utils.data.preprocess import process_PAM
from txai.utils.predictors import eval_mvts_transformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PAMDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

test_f1 = []
elapsed = []

for i in range(1, 6):
    start = time.time()
    print(f'\n------------------ Split {i} ------------------')
    trainPAM, val, test = process_PAM(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/PAMAP2data/', gethalf = True)
    # Output of above are chunks
    train_dataset = PAMDataset(trainPAM.X, trainPAM.time, trainPAM.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

    model = TransformerMVTS(
        d_inp = trainPAM.X.shape[2],
        max_len = trainPAM.X.shape[0],
        n_classes = 8,
    )

    # Convert to GPU:
    model.to(device)

    spath = f'models/transformer_split={i}.pt'

    # Train model:
    model, loss, auc = train(model, train_loader, 
        val_tuple = (val.X, val.time, val.y), n_classes = 8, num_epochs = 100,
        save_path = spath, validate_by_step = None)

    elapsed.append(time.time() - start)

    # Get test result:
    f1 = eval_mvts_transformer((test.X, test.time, test.y), model, batch_size = 64)
    test_f1.append(f1)
    print('Test F1: {:.4f} \t Time: {}'.format(f1, elapsed[-1]))

print('='*50)
print('Testing Scores:', test_f1)
print('Avg: {:.4f}'.format(np.mean(test_f1)))
print('Avg elapsed: {:.4f}'.format(np.mean(elapsed)))