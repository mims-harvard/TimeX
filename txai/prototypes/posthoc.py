import torch
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import GridSearchCV

@torch.no_grad()
def find_dbscan_ptypes(z_train, z_test):
    '''
    Finds clusters using dbscan and returns cluster labels
    '''

    z_mask = F.normalize(z_train, dim = 1).squeeze()
    z_mask_test = F.normalize(z_test, dim = 1).squeeze()

    train_size = z_mask.shape[0]

    z_mask = torch.cat([z_mask, z_mask_test], dim = 0)

    # Compute distances (cosine distances b/c normalized):
    zm_dist_pt = (torch.matmul(z_mask, z_mask.transpose(0, 1)) + 1) / 2.0
    
    zm_dist = zm_dist_pt.detach().clone().cpu().numpy()

    # Find eps as 80th percentile of distance distribution:
    L = zm_dist_pt.flatten().shape[0]
    si = torch.randperm(L)[:(L // 100)]
    eps = torch.quantile(zm_dist_pt.flatten()[si], 5e-3, interpolation = 'linear').item()

    best_DB = DBSCAN(eps = eps, metric = 'precomputed')

    # searcher = GridSearchCV(DB, pgrid, verbose = 2)
    # searcher.fit(zm_dist)

    # best_DB = searcher.best_estimator_

    clusters = best_DB.fit_predict(zm_dist)

    # Separate out train and test:
    clusters_train = clusters[:train_size]
    clusters_test = clusters[train_size:]

    return clusters_train, clusters_test

@torch.no_grad()
def find_kmeans_ptypes(z_train, z_test, n_clusters = 25):
    '''
    Finds clusters using dbscan and returns cluster labels
    '''

    z_mask = F.normalize(z_train, dim = 1).squeeze()
    z_mask_test = F.normalize(z_test, dim = 1).squeeze()

    train_size = z_mask.shape[0]

    z_mask = torch.cat([z_mask, z_mask_test], dim = 0)

    # Compute distances (cosine distances b/c normalized):
    zm_dist_pt = (torch.matmul(z_mask, z_mask.transpose(0, 1)) + 1) / 2.0
    
    zm_dist = zm_dist_pt.detach().clone().cpu().numpy()

    # k-means

    clusters = KMeans(n_clusters = n_clusters, verbose = 2, metric = 'precomputed').fit_predict(zm_dist)

    # Separate out train and test:
    clusters_train = clusters[:train_size]
    clusters_test = clusters[train_size:]

    return clusters_train, clusters_test

def find_nearest_explanations(z_query, z_ref, dist = 'cosine', n_exps_per_q = 5):

    if dist == 'cosine':
        
        zq_norm = F.normalize(z_query, dim = 1)
        zr_norm = F.normalize(z_ref, dim = 1)

        print('zq_norm', zq_norm.shape)
        print('zr_norm', zr_norm.shape)

        sims = torch.matmul(zq_norm, zr_norm.transpose(0, 1))

    else:
        raise ValueError('{} metric not implemented'.format(dist))

    best_inds = sims.argsort(dim = -1, descending = True)[:,:n_exps_per_q]

    return best_inds # Shape (nq, n_exps_per_q)

def filter_prototypes(p_z, z_ref, dist = 'cosine', lower_bound = 5, get_count_dist = False):

    if dist == 'cosine':
        
        zq_norm = F.normalize(p_z, dim = 1)
        zr_norm = F.normalize(z_ref, dim = 1)

        print('zq_norm', zq_norm.shape)
        print('zr_norm', zr_norm.shape)

        sims = torch.matmul(zq_norm, zr_norm.transpose(0, 1))

    else:
        raise ValueError('{} metric not implemented'.format(dist))

    am = sims.argmax(dim=0)
    #print('am', am)

    # Find all inds that have at least lower_bound occurences:
    found = torch.unique(am)
    found_counts = torch.tensor([(f == am).sum() for f in found])
    count_mask = found_counts > lower_bound

    choices = found[count_mask]

    print('choices', choices)

    if get_count_dist:
        count_dist_overall = torch.tensor([(f == am).sum() for f in torch.arange(p_z.shape[0])])
        return choices, count_dist_overall
    else:
        return choices

    
