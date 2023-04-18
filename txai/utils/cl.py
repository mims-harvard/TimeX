import torch
import numpy as np
import torch.nn.functional as F

def basic_negative_sampling(batch, batch_ids, dataX, num_negatives):
    '''
    batch: (B, T, d)
    batch_ids: ()
    dataX: (T, Nx, d)
    num_negatives: int

    output: (B, num_negatives) - gives ints
    '''

    mask = torch.randn(batch.shape[0], dataX.shape[1]) # Size (B, Nx)
    inds = torch.empty(batch.shape[0], num_negatives).long()
    for i, bid in enumerate(batch_ids):
        mask[i, bid] = -1e9 # Effectively ignoring
        inds[i,:] = mask[i,:].topk(k=num_negatives)[1] # Get indices

    # # randn, get top-k

    # #possible = mask.nonzero(as_tuple=True)[0].numpy()
    # inds = torch.from_numpy(np.random.choice(possible, size = (num_negatives,)))

    # mask = torch.zeros(dataX.shape[1]).bool(); mask[inds] = 1

    return inds


@torch.no_grad() # No grad so gradients aren't carried into similarity computations on batch
def in_batch_triplet_sampling(z_main, num_triplets_per_sample = 1):
    '''
    Samples triplets from the batch and separates into anchors, positives, and negatives based 
        on reference embedding similarity
    '''

    z_main_cpu = z_main.detach().clone().cpu()

    # Get two rows of unique indices:
    B, d = z_main_cpu.shape
    anchor_inds = torch.arange(B)

    pmat = (np.ones((B, B)) - np.eye(B)) / (B - 1)

    all_samps_tensors = []
    all_anchor_inds = []

    for i in range(num_triplets_per_sample):

        samps = [np.random.choice(B, size = (2,), replace = True, p = pmat[i,:]) for i in range(B)]

        samps_mat = np.stack(samps, axis = 0)
        samps_tensor = torch.from_numpy(samps_mat).long()
        all_samps_tensors.append(samps_tensor)
        all_anchor_inds.append(anchor_inds.clone())

    samps_tensor = torch.cat(all_samps_tensors, dim = 0)
    anchor_inds = torch.cat(all_anchor_inds).flatten()

    # Calculate similarities and get masks
    # Use euclidean distance bc this is default in triplet loss pytorch for now
    leftside = (z_main_cpu[anchor_inds] - z_main_cpu[samps_tensor[:,0]]).norm(p=2, dim = 1)
    rightside = (z_main_cpu[anchor_inds] - z_main_cpu[samps_tensor[:,1]]).norm(p=2, dim = 1)

    left_larger = (leftside > rightside)

    # Assign respective indices to each side
    positives = torch.where(~left_larger, samps_tensor[:,0], samps_tensor[:,1])
    negatives = torch.where(left_larger, samps_tensor[:,0], samps_tensor[:,1]) # Larger distance is negative

    return anchor_inds, positives, negatives
