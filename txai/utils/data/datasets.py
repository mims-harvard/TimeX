import torch

class DatasetwInds(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X
        self.times = times
        self.y = y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y, torch.tensor(idx).long().to(x.device)