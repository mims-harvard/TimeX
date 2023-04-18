import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1] # Number of batches

        P_time = P_time.float()

        # timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales).to(device) this was numpy
        timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales).to(device)

        #times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        times = P_time.unsqueeze(2)

        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        #pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        #pe = pe.to(device)
        return pe