import torch
import torch.nn.functional as F

from txai.utils.predictors.loss_cl import LabelConsistencyLoss

# Old selection criteria, here for reference: -----------------
def lower_bound_performance(lower_bound):
    def func(metric, sparsity):
        if metric >= lower_bound:
            return (1 - sparsity)
        return 0
    
    return func

def best_metric():
    def func(metric, sparsity):
        return metric
    return func
# -------------------------------------------------------------

def cosine_sim(out_dict, val = None):
    full_z, mask_z = out_dict['all_z']
    sim = F.cosine_similarity(full_z, mask_z, dim = -1)
    return sim.mean().detach().cpu().item()

def small_mask(out_dict, val = None):
    mask = out_dict['ste_mask']
    return -1.0 * mask.float().detach().cpu().mean().item()


def sim_small_mask(out_dict, val = None):
    return cosine_sim(out_dict) + small_mask(out_dict)

def simloss_on_val_wboth(sim_criterion, lam = 1.0):
    # Early stopping for sim loss

    def f(out_dict, val = None):
        org_z, con_z = out_dict['all_z']
        mlab, flab = out_dict['pred_mask'], out_dict['pred']
        L = sim_criterion[0](org_z, con_z) + lam * sim_criterion[1](mlab, flab)
        return -1.0 * L # Need maximum, so return negative

    return f
