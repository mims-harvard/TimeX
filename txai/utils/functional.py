import torch
from torch import Tensor
import torch.nn.functional as F

def mahalanobis_dist(z, mu, sigma_inv):

    # Repeat mu for batching
    mu_rep = mu.unsqueeze(0).repeat(z.shape[0], 1)
    sigma_inv_rep = sigma_inv.unsqueeze(0).repeat(z.shape[0], 1, 1)

    assert z.shape == mu_rep.shape, 'Shape mismatch, z=({}), mu=({})'.format(z.shape, mu_rep.shape)

    delta = (z - mu_rep).unsqueeze(-1)
    # print('delta', delta.shape)
    # print('sigma', sigma_inv_rep.shape)
    # print('bmm size', torch.bmm(sigma_inv_rep, delta).shape)
    m = torch.bmm(delta.transpose(1, 2), torch.bmm(sigma_inv_rep, delta))
    return m.sqrt()

def transform_to_attn_mask(linear_mask):
    '''
    NOTE: Assumes that input is STE, i.e. binary
    NOTE: If linear_mask is multivariate, expects (T, B, d) size
    '''

    if len(linear_mask.shape) > 2:
        # If multivariate mask, convert to univariate:
        # Needs to be differentiable
        linear_mask = F.hardtanh(linear_mask.sum(dim=-1)).transpose(0, 1) # Hard tanh sends anything >= 1 to 1, stil differentiable
        # 0's stay as 0's

    attn_mask = linear_mask.unsqueeze(-1).repeat(1, 1, linear_mask.shape[1])
    attn_mask = attn_mask * attn_mask.transpose(1, 2) # Flip and multiply
    return attn_mask

def js_divergence(p: Tensor, q: Tensor) -> Tensor:
    # JSD(P || Q)
    # Assumes both have alread
    # Implementation borrowed from: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/divergence.html
    m = 0.5 * (p + q)
    return (0.5 * F.kl_div(p.log(), m) + 0.5 * F.kl_div(q.log(), m))

def cosine_sim_matrix(z1, z2):
    # Shape: (B1, d_z), (B2, d_z)
    z1 = F.normalize(z1, dim = -1)
    z2 = F.normalize(z2, dim = -1)

    return torch.matmul(z1, z2.transpose(0, 1)) # (B, B) matrix

def gs_reparameterize(total_mask_probs, tau = 1.0, use_ste = True):

        #if total_mask.shape[-1] == 1:
        # Need to add extra dim:
        inv_probs = 1 - total_mask_probs
        total_mask_prob = torch.cat([inv_probs, total_mask_probs], dim=-1)

        # else:
        #     total_mask_prob = total_mask.softmax(dim=-1)

        if use_ste:
            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = tau, hard = True)[...,1]
        else:
            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = tau, hard = False)[...,1]

        return total_mask_reparameterize

def dkl_bernoullis(p1, p2):
    d = p1 * torch.log(p1 / p2 + 1e-9) + (1 - p1) * torch.log((1 - p1) / (1 - p2) + 1e-9)
    if torch.any(d.isnan()):
        print('NaN in DKL')
    # print(p1)
    # print(p2)
    return d