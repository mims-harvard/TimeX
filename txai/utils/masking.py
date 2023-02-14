import torch

def gaussian_time_samples(X, perturbation_freq = 0.2):
    '''
    Samples out time points according to some frequency and perturbs every sensor
        at those chosen time points (with Gaussian noise).
    '''

    squeeze_back = False
    if len(X.shape) > 2:
        X = X.squeeze() # Should squeeze out batch dimension
        squeeze_back = True

    # Choose N time points

    time_samp = torch.rand((X.shape[0],))

    bool_samp = (time_samp < perturbation_freq)

    # At selected times, add Gaussian noise equivalent to feature-wide statistics:

    mu = torch.mean(X, dim=-1)
    std = torch.std(X, dim=-1)

    # Sample and mask out those time points that we don't perturb
    noise = [torch.normal(mu, std) for i in range(X.shape[0]) if bool_samp[i]]
    noise = torch.stack(noise) # Stack together normal noises
    
    Xpert = torch.where(bool_samp, noise, X) 
    # Place noise where bool_samp is true, X where it's not

    if squeeze_back:
        Xpert = Xpert.unsqueeze(dim=1)

    return Xpert

def random_time_mask(rate, size):
    '''
    Assumes the size is (T,d)
    ''' 

    size = tuple(size)
    mask = torch.zeros(size[0])
    n = int(size[0] * rate)
    inds = torch.randperm(size[0])[:n]
    mask[inds] = 1
    mask = mask.unsqueeze(-1).repeat(1,size[1]) # Repeat along time dimensions

    return mask

def dyna_norm_mask(Xtrain):
    # Returns a function that, when called, gives a dynamic normal mask application

    # Compute mean, std:
    std = Xtrain.std(unbiased = True, dim = 0)
    mu = Xtrain.mean(dim=0)

    def apply_mask(X, mask):
        to_replace = (mu + torch.randn_like(std) * std).unsqueeze(0).repeat(X.shape[0], 1, 1)
        return (mask * X) + (1 - mask) * to_replace

    return apply_mask

