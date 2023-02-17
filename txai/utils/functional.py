import torch

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