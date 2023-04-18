import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F

sqrt2PI = math.sqrt(2.0 * math.pi)

class Smoother(nn.Module):
    def __init__(self, memory_efficient = False, init_p = None):
        super(Smoother, self).__init__()
        # Parameterize smoothing length s.t. it isn't so close to 0
        if init_p is None:
            self.p = nn.Parameter(torch.rand(1) * 4, requires_grad = True)
        else:
            self.p = nn.Parameter(torch.tensor(init_p), requires_grad = True)
        self.memory_efficient = memory_efficient
        # MUST PARAMETERIZE TO BE ABSOLUTE VALUE

    def generate_spread_coefs(self, times, tcenter):
        # tcenter = centered point in the curve:
        # Times should already be masked
        return 1.0 / (self.p * sqrt2PI) * torch.exp(-0.5 * ((times - tcenter) / self.p))

    def forward(self, src, time, mask = None):
        '''
        Mask determines level of repeating:
        Need to do in a batched fashion
        src: (T, B, 1) Src should be single-channeled
        time: (T, B)
        mask: (T, B, 1) mask follows size of src 
        '''

        new_src = torch.empty_like(src).to(src.device)

        if self.memory_efficient:
            
            for t in range(times.shape[0]):
                coef = self.generate_spread_coefs(time, tcenter = time[t])

                if mask is not None:
                    coef = coef * mask

                coef = coef.softmax(dim=0) # Softmax across time dimension
                print('Coef', coef)
                exit()

                new_src[t,:,0] = (src * coef).sum(dim=0) # Should be (B,) tensor 

        else: # TODO: implement version that uses more memory but consists of only tensor operations (no for loop)
            # Generate (T, T, B) matrix of coefficients
            pass

        return new_src

# As functions:

def generate_spread_coefs(p, times, tcenter):
    '''
    tcenter = centered point in the curve:
    Times should already be masked
    p should be (B,)
    '''
    #print('tcenter', tcenter.shape)
    tcenter_rep = tcenter.unsqueeze(0).repeat(times.shape[0], 1)
    p_rep = p.repeat(1, times.shape[0]).transpose(0, 1)
    # print('times', times.shape)
    # print('tcenter_rep', tcenter_rep.shape)
    # print('prep', p_rep.shape)
    return 1.0 / (p_rep * sqrt2PI + 1e-9) * torch.exp(-0.5 * ((times - tcenter_rep) / (p_rep + 1e-9)) ** 2)

def smoother(src, time, p, mask = None):
        new_src = torch.empty_like(src).to(src.device)

        #print('time', time.shape)
            
        for t in range(time.shape[0]):
            # time: (B, T)
            # tcenter: (B,)

            coef = generate_spread_coefs(p = p, times = time, tcenter = time[t,:])

            coef = coef[:(t+1),:].softmax(dim=0) # Softmax across time dimension

            if mask is not None:
                coef = coef * mask[:,:(t+1)].transpose(0, 1).squeeze() # Mask after softmax
                #coef = coef * mask.transpose(0, 1).squeeze()

            #coef = coef[:(t+1),:] / (coef[:(t+1),:].sum() + 1e-9)

            #print('coef', coef.shape)

            #coef_rep = coef.repeat(1, src.shape[1])
            new_src[t,:,0] = (src[:(t+1),:,0] * coef).sum(dim=0) # Should be (B,) tensor 

        # else: # TODO: implement version that uses more memory but consists of only tensor operations (no for loop)
        #     # Generate (T, T, B) matrix of coefficients
        #     pass

        return new_src

def exponential_smoother(src, time, p):
    '''
    Simple exponential smoother on whole sample

    NOTE: only works on single-channel, but simple to extend to multivariate

    p: (B,) tensor
    '''

    #print('p', p.shape)

    T, B, d = src.shape

    #coef = torch.zeros(T, T, B, device = src.device)

    #print('p', p)

    coef = torch.cat([torch.eye(T, device = src.device).unsqueeze(-1) * p[i] for i in range(B)], dim = -1)

    #print('coef 1', coef)

    coef[0,0,:] = 1 # Top left is 1

    for i in range(T):
        restcol = ((1 - p).repeat(1,T - i).transpose(0,1) ** torch.arange(T - i, device = p.device).unsqueeze(-1).repeat(1,B))
        #print('rc', restcol.shape)
        coef[i,i:,:] = coef[i,i,:].unsqueeze(0).repeat(T-i,1) * restcol

    #print('coef', coef)
    smoothed_src = torch.bmm(coef.permute(2, 0, 1), src.transpose(1, 0))

    #print('Smoothed', smoothed_src.shape)

    return smoothed_src.transpose(0, 1)


if __name__ == '__main__':
    # TESTING

    import matplotlib.pyplot as plt

    # Try smoothing operator + visualize
    rand_seq = torch.randn(10, 1, 1).abs() * torch.arange(10).unsqueeze(-1).unsqueeze(-1)

    texp = torch.arange(rand_seq.shape[0]).unsqueeze(-1).repeat(1, rand_seq.shape[1])
    print('t', texp)
    smoothed = smoother(rand_seq, time = texp, p = 4)
    print('Before smooth', rand_seq.squeeze())
    print('After smooth', smoothed.squeeze())

    rnp = rand_seq.squeeze().numpy()
    snp = smoothed.squeeze().numpy()

    plt.plot(np.arange(10), rnp)
    plt.plot(np.arange(10), snp)
    plt.hlines(np.mean(rnp), 0, 10, linestyles='dashdot', color = 'green')
    plt.show()

