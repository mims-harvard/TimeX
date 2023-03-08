import torch
import torch.nn as nn
import torch.nn.functional as F

from txai.models.complex_layers import ComplexLinear, ComplexReLU

class SpectralConv1d(nn.Module):
    def __init__(self, modes1, hidden_channels = 64):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform    
        """
        self.hidden_channels = hidden_channels
        self.modes1 = modes1
        #Number of Fourier modes to multiply, at most floor(N/2) + 1

        # Create complex-valued layers: https://discuss.pytorch.org/t/complex-valued-neural-network/117090
        #self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, 1, self.modes1, dtype=torch.cfloat))
        self.fc1 = ComplexLinear(self.modes1, hidden_channels)
        self.complex_relu = ComplexReLU()
        self.fc2 = ComplexLinear(hidden_channels, self.modes1)
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # def forward(self, x):
    #     batchsize = x.shape[0]
    #     #Compute Fourier coeffcients up to factor of e^(- something constant)
    #     #x = torch.cos(x)
    #     x_ft = torch.fft.rfft(x,norm='ortho')
    #     modes1 = x.size(0)//2 + 1
    #     out_ft = torch.zeros(batchsize, 1, modes1,  device=x.device, dtype=torch.cfloat)
    #     out_ft = self.compl_mul1d(x_ft[:, :, :modes1], self.weights1) 
    #     r = out_ft[:, :, :modes1].abs()
    #     p = out_ft[:, :, :modes1].angle() 
    #     # return torch.concat([r,p],-1), out_ft
    #     return r, out_ft

    def forward(self, src):
        x_ft = torch.fft.rfft(src, dim = 0, norm='ortho').squeeze().transpose(0, 1)
        z_ft = self.complex_relu(self.fc1(x_ft))
        out_ft = self.fc2(z_ft).abs()
        return out_ft, x_ft

class FourierMasker(nn.Module):
    def __init__(self, maxlen, hidden_channels = 64):
        super(FourierMasker, self).__init__()
        
        self.maxlen = maxlen
        self.scorer = SpectralConv1d(maxlen // 2 + 1, hidden_channels = hidden_channels)

    def forward(self, src):
        
        logits, x_ft = self.scorer(src)
        probs = logits.sigmoid()

        if self.training:
            # Through Gumbel reparameterization
            full_probs = torch.cat([(1 - probs).unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)
            sampled = F.gumbel_softmax(torch.log(full_probs + 1e-9), hard = True, tau = 1)
            # print('sampled', sampled.shape)
            # exit()    
            mask = sampled[...,1]
        else:
            mask = (probs > 0.5).float()

        # Mask values, get irfft:
        print('mask', mask)
        mask_x_ft = x_ft * mask
        transform_x = torch.fft.irfft(mask_x_ft, n = self.maxlen, norm = 'ortho')
        transform_x = transform_x.unsqueeze(-1).transpose(0, 1) # Change back to single-channel and (T, B, 1)

        # output mask
        return transform_x, logits, mask