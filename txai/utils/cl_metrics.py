from tqdm import trange
import numpy as np

import torch
import torch.nn.functional as F

def sim_mat(full_z, mask_z):
    '''
    Calculates similarity matrix bw all samples in embeddings

    NOTE: Very inefficient, don't use if you have a lot of embeddings
    '''
    out_norm = F.normalize(full_z, dim = -1)
    out_masked_norm = F.normalize(mask_z, dim = -1)

    mat = np.zeros((out_norm.shape[0], out_masked_norm.shape[0]))
    for i in trange(out_norm.shape[0]):
        for j in range(i, out_masked_norm.shape[0]):
            mat[i,j] = torch.dot(out_norm[i,:], out_masked_norm[j,:])

    return mat