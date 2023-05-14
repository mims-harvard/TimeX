import torch
import numpy as np
import matplotlib.pyplot as plt
from txai.models.modelv6_v2 import Modelv6_v2
from txai.vis.vis_saliency import vis_one_saliency, vis_one_saliency_univariate

def get_x_mask_borders(mask):
    nz = mask.nonzero(as_tuple=True)[0].tolist()
    # Get contiguous:
    nz_inds = [(nz[i], nz[i+1]) for i in range(len(nz) - 1) if (nz[i] == (nz[i+1] - 1))]
    return nz_inds

def plot_heatmap(mask_logits, smooth_src, ax, fig):
    x_range = torch.arange(smooth_src.shape[0])
    px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(smooth_src), max(smooth_src)])
    #ax[0,i].imshow(mask_logits[i,...].T, alpha = 0.5, cmap = 'Greens')
    cmap = ax.pcolormesh(px, py, mask_logits.T, alpha = 0.5, cmap = 'Greens')
    fig.colorbar(cmap, ax = ax)

def vis_concepts(model, test_tup, show = True):

    src, times, y = test_tup

    found_masks, found_smooth_src = model.get_concepts(src, times, captum_input = False)

    found_masks_np = [found_masks[i].cpu().numpy() for i in range(len(found_masks))]
    found_smooth_src_np = [found_smooth_src[i].cpu().numpy() for i in range(len(found_smooth_src))]

    fig, ax = plt.subplots(1, model.n_concepts)

    for i in range(model.n_concepts):
        ax[i].plot(found_smooth_src_np[i][...,0], color = 'black')
        ax[i].set_title('Concept {}'.format(i))
        plot_heatmap(np.expand_dims(found_masks_np[i], axis = 1), found_smooth_src_np[i][...,0], ax[i], fig)

    if show:
        plt.show()

def vis_prototypes(model, test_tup, show = True, k = 3):

    src, times, y = test_tup

    with torch.no_grad():
        found_masks, found_smooth_src = model.get_concepts(src, times, captum_input = False, k = k)

    found_masks_np = [found_masks[i].cpu().numpy() for i in range(len(found_masks))]
    found_smooth_src_np = [found_smooth_src[i].cpu().numpy() for i in range(len(found_smooth_src))]

    fig, ax = plt.subplots(k, model.n_prototypes, figsize = (7, 20), dpi = 200)

    for i in range(model.n_prototypes):
        for j in range(k):
            ax[j, i].plot(found_smooth_src_np[i][:,j,0], color = 'black')
            #ax[j, i].set_title('Prototype {}'.format(i))
            plot_heatmap(np.expand_dims(found_masks_np[i][j,:], axis = 1), found_smooth_src_np[i][:,j,0], ax[j, i], fig)

    if show:
        plt.show()

def vis_sim_to_ptypes(X_nearby_list, mask_nearby_list, y_nearby_list = None, show = True):

    Nq = len(X_nearby_list)

    #plt.figure(dpi=200)
    fig, ax = plt.subplots(X_nearby_list[0].shape[1], Nq, sharex = True, dpi = 200)

    xr = np.arange(X_nearby_list[0].shape[0])

    for i in range(Nq):

        Xq_ref = X_nearby_list[i]
        mask_ref = mask_nearby_list[i]
        if y_nearby_list is not None:
            y_ref = y_nearby_list[i]

        if y_nearby_list is not None:
            ax[0,i].set_title('Example {}, label = {}'.format(i, y_ref[0]))
        else:
            ax[0,i].set_title('Example {}'.format(i))

        for j in range(Xq_ref.shape[1]):
            mq_ij = np.expand_dims(mask_ref[j,:], axis = 1)
            print('mq', mq_ij.shape)

            ax[j,i].plot(xr, Xq_ref[:,j,:], color = 'black')
            if y_nearby_list is not None:
                ax[j,i].set_title(f'label = {y_ref[j]}', fontdict = {'fontsize':10})
            plot_heatmap(mq_ij, Xq_ref[:,j,:], ax = ax[j,i], fig = fig)

    if show:
        plt.tight_layout(pad=0.3)
        plt.show()

def vis_exps_w_sim(X_query, mask_query, X_nearby_list, mask_nearby_list, show = True):

    Nq = X_query.shape[1]

    #plt.figure(dpi=200)
    fig, ax = plt.subplots(X_nearby_list[0].shape[1] + 1, Nq, sharex = True, dpi = 200)

    xr = np.arange(X_query.shape[0])

    for i in range(Nq):

        Xq_ref = X_nearby_list[i]
        mask_ref = mask_nearby_list[i]

        ax[0,i].plot(xr, X_query[:,i,:], color = 'black')
        # Plot heatmap:
        plot_heatmap(np.expand_dims(mask_query[i,:], axis = -1), X_query[:,i,:], ax = ax[0,i], fig = fig)
        ax[0,i].set_title('Example {}'.format(i))

        for j in range(Xq_ref.shape[1]):
            mq_ij = np.expand_dims(mask_ref[j,:], axis = 1)
            print('mq', mq_ij.shape)

            ax[(j+1),i].plot(xr, Xq_ref[:,j,:], color = 'black')
            plot_heatmap(mq_ij, Xq_ref[:,j,:], ax = ax[(j+1),i], fig = fig)

    if show:
        plt.show()

def logical_or_mask_along_explanations(total_mask):
    tmask = (total_mask.sum(dim=-1) > 0).float() # Effectively ORs along last dimension
    return tmask

def visualize_explanations_new(model, test_tup, n = 3, class_num = None, show = True, heatmap = True, topk = None, seed = None):
    '''
    TODO: Rewrite

    - Shows each extracted explanations along with importance scores for n samples
    - TODO in future: aggregate multiple explanation types into one visualization

    NOTE: Only works for regular time series
    '''
    # Quick function to visualize some samples in test_tup
    # FOR NOW, assume only 2 masks, 2 concepts

    X, times, y = test_tup

    d_z = X.shape[-1]

    choices = np.arange(X.shape[1])
    if class_num is not None:
        choices = choices[(y == class_num).cpu().numpy()]
    np.random.seed(seed)
    inds = torch.from_numpy(np.random.choice(choices, size = (n,), replace = False)).long()
    #if isinstance(model, Modelv6_v2) or isinstance(model, Modelv6_v2_concepts):

    sampX, samp_times, samp_y = X[:,inds,:], times[:,inds], y[inds]
    x_range = torch.arange(sampX.shape[0])

    model.eval()
    with torch.no_grad():
        out = model(sampX, samp_times, captum_input = False)
        #pred, pred_mask, mask_in, ste_mask, smoother_stats, smooth_src = model(sampX, samp_times, captum_input = False)

    pred, pred_mask = out['pred'], out['pred_mask']
    mask_logits = out['mask_logits']

    #smooth_src = torch.stack(out['smooth_src'], dim = 0) # Shape (N_c, T, B, d)
    smooth_src = out['smooth_src'].unsqueeze(0)

    pred = pred.softmax(dim=1).argmax(dim=1)
    pred_mask = pred_mask.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)

    title_format1 = 'y={:1d}, yhat={:1d}'

    fig, ax = plt.subplots(d_z, n, sharex = True, squeeze = False)

    for i in range(n):

        if sampX.shape[-1] == 1:
            # print('ML', mask_logits.shape)
            # exit()
            vis_one_saliency_univariate(sampX[:,i,:], mask_logits[i,:,:].transpose(0,1), ax = ax[0,i], fig = fig)
        else:
            vis_one_saliency(sampX[:,i,:], mask_logits[:,i,:], ax = ax, fig = fig, col_num = i)

    if show:
        plt.show()

def visualize_explanations(model, test_tup, n = 3, class_num = None, show = True, heatmap = True, topk = None, seed = None):
    '''
    TODO: Rewrite

    - Shows each extracted explanations along with importance scores for n samples
    - TODO in future: aggregate multiple explanation types into one visualization

    NOTE: Only works for regular time series
    '''
    # Quick function to visualize some samples in test_tup
    # FOR NOW, assume only 2 masks, 2 concepts

    X, times, y = test_tup

    choices = np.arange(X.shape[1])
    if class_num is not None:
        choices = choices[(y == class_num).cpu().numpy()]
    np.random.seed(seed)
    inds = torch.from_numpy(np.random.choice(choices, size = (n,), replace = False)).long()
    #if isinstance(model, Modelv6_v2) or isinstance(model, Modelv6_v2_concepts):
    num_on_x = (2)
    # else:
    #     num_on_x = 1 + model.n_concepts
    fig, ax = plt.subplots(num_on_x, n, sharex = True)

    sampX, samp_times, samp_y = X[:,inds,:], times[:,inds], y[inds]
    x_range = torch.arange(sampX.shape[0])

    model.eval()
    with torch.no_grad():
        out = model(sampX, samp_times, captum_input = False)
        #pred, pred_mask, mask_in, ste_mask, smoother_stats, smooth_src = model(sampX, samp_times, captum_input = False)

    pred, pred_mask = out['pred'], out['pred_mask']
    masks = (out['ste_mask']).float() # All masks
    print('masks', masks.shape)
    mask_logits = out['mask_logits'].detach().cpu().numpy()
    print('mask_logits', mask_logits.shape)
    aggregate_mask_discrete = logical_or_mask_along_explanations(masks)
    aggregate_mask_continuous = mask_logits.sum(-1)

    #smooth_src = torch.stack(out['smooth_src'], dim = 0) # Shape (N_c, T, B, d)
    smooth_src = out['smooth_src'].unsqueeze(0)

    pred = pred.softmax(dim=1).argmax(dim=1)
    pred_mask = pred_mask.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)

    title_format1 = 'y={:1d}, yhat={:1d}'

    for i in range(n): # Iterate over samples

        # fit lots of info into the title
        yi = samp_y[i].item()
        pi = pred[i].item()
        ax[0,i].set_title(title_format1.format(yi, pi))

        # Top plot shows full sample with full mask:
        sX = sampX[:,i,:].cpu().numpy()
        # Stays fixed (for grids on samples):
        px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(sX), max(sX)])
        ax[0,i].plot(x_range, sX, color = 'black')


        if heatmap: # Plot discrete mask
            if topk is not None:
                tk_inds = np.flip(np.argsort(aggregate_mask_continuous[i,...]))[:topk]
                tk_mask = np.zeros_like(aggregate_mask_continuous[i,...])
                tk_mask[tk_inds] = 1
                block_inds = get_x_mask_borders(mask = torch.from_numpy(tk_mask))
                for k in range(len(block_inds)):
                    ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
            else:
                cmap = ax[0,i].pcolormesh(px, py, np.expand_dims(aggregate_mask_continuous[i,...], -1).T, alpha = 0.5, cmap = 'Greens')
                fig.colorbar(cmap, ax = ax[0,i])
        else:
            block_inds = get_x_mask_borders(mask = aggregate_mask_discrete[i,...]) # GET WHOLE MASK
            for k in range(len(block_inds)):
                ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

        # cscores = out['concept_scores']#.squeeze() # (B, Nc)
        # cselect_inds = out['concept_selections_inds']
        # print('cselect_inds\n', cselect_inds)
        #exit()

        for j in range(num_on_x-1):

            # Add subtitles:
            #ax[(j+1)][i].set_title('a={}, p={}'.format())
            ax[(j+1)][i].plot(x_range, smooth_src[j,:,i,:].cpu().numpy(), color = 'black')

            #print('cscores', cscores)
            #ax[(j+1)][i].set_title('score = {:.4f}'.format(cscores[i,j].detach().clone().item()))

            if heatmap:
                if topk is not None:
                    # Mask in those that are in the top-k
                    tk_inds = np.flip(np.argsort(mask_logits[i,:,j]))[:topk]
                    tk_mask = np.zeros_like(mask_logits[i,:,j])
                    tk_mask[tk_inds] = 1
                    block_inds = get_x_mask_borders(mask = torch.from_numpy(tk_mask))
                    for k in range(len(block_inds)):
                        ax[j+1][i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
                else:
                    cmap = ax[j+1][i].pcolormesh(px, py, np.expand_dims(mask_logits[i,:,j], -1).T, alpha = 0.5, cmap = 'Greens')
                    fig.colorbar(cmap, ax = ax[j+1][i])

            else:
                block_inds = get_x_mask_borders(mask = masks[i,:,j])
                for k in range(len(block_inds)):
                    ax[j+1][i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

    if show:
        plt.show()