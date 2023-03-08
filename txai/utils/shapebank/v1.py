import torch
import numpy as np

def gen_sample(template, increase = True):

    length = np.random.choice(np.arange(start=5, stop=45))
    if increase == True:
        seq = np.linspace(-2, 2, num = int(length))
    else:
        seq = np.linspace(2, -2, num = int(length))

    seq *= np.random.normal(1.0, scale = 0.01, size = seq.shape)

    # Get mask w/sampled location:
    loc = np.random.choice(np.arange(start=0, stop=int(template.shape[0]-length)))

    a = torch.randn_like(template)
    a[loc:(loc+length),0,0] = torch.from_numpy(seq)

    return a

def gen_sample_zero(template, increase = True):

    length = np.random.choice(np.arange(start=5, stop=45))
    amp = np.random.normal(1.0, scale = 0.25)
    if increase == True:
        seq = np.linspace(-2, 2, num = int(length))
    else:
        seq = np.linspace(2, -2, num = int(length))

    seq *= np.random.normal(1.0, scale = 0.05, size = seq.shape)

    # Get mask w/sampled location:
    loc = np.random.choice(np.arange(start=0, stop=int(template.shape[0]-length)))

    a = torch.zeros_like(template)
    a[loc:(loc+length),0,0] = torch.from_numpy(seq)

    return a

def gen_dataset(template, samps = 1000, device = None):
    inc = torch.cat([gen_sample(template, increase = True) for _ in range(samps)], dim = 1).to(device)
    dec = torch.cat([gen_sample(template, increase = False) for _ in range(samps)], dim = 1).to(device)
    
    times = torch.arange(inc.shape[0]).unsqueeze(-1).repeat(1, samps * 2).to(device)
    whole = torch.cat([inc, dec], dim=1).to(device)
    batch_id = torch.cat([torch.zeros(inc.shape[1]), torch.ones(dec.shape[1])]).to(device).long()
    return whole, times, batch_id

def gen_dataset_zero(template, samps = 1000, device = None):
    inc = torch.cat([gen_sample_zero(template, increase = True) for _ in range(samps)], dim = 1).to(device)
    dec = torch.cat([gen_sample_zero(template, increase = False) for _ in range(samps)], dim = 1).to(device)
    
    times = torch.arange(inc.shape[0]).unsqueeze(-1).repeat(1, samps * 2).to(device)
    whole = torch.cat([inc, dec], dim=1)
    batch_id = torch.cat([torch.zeros(inc.shape[1]), torch.ones(dec.shape[1])]).to(device).long()
    return whole, times, batch_id