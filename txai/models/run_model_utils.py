import torch

def concat_all_dicts(dlist, org_v = False):
    # Marries together all dictionaries
    # Will change based on output from model

    mother_dict = {k:[] for k in dlist[0].keys()}

    is_tensor_list = []

    for d in dlist:
        for k in d.keys():
            if k == 'smooth_src' and org_v:
                mother_dict[k].append(torch.stack(d[k], dim = -1))
            else:   
                mother_dict[k].append(d[k])

    mother_dict['pred'] = torch.cat(mother_dict['pred'], dim = 0).cpu()
    mother_dict['pred_mask'] = torch.cat(mother_dict['pred_mask'], dim = 0).cpu()
    mother_dict['mask_logits'] = torch.cat(mother_dict['mask_logits'], dim = 0).cpu()
    if org_v:
        mother_dict['concept_scores'] = torch.cat(mother_dict['concept_scores'], dim = 0).cpu()
    mother_dict['ste_mask'] = torch.cat(mother_dict['ste_mask'], dim = 0).cpu()
    # [[(), ()], ... 24]
    mother_dict['smooth_src'] = torch.cat(mother_dict['smooth_src'], dim = 1).cpu() # Will be (T, B, d, ne)

    L = len(mother_dict['all_z'])
    mother_dict['all_z'] = (
        torch.cat([mother_dict['all_z'][i][0] for i in range(L)], dim = 0).cpu(), 
        torch.cat([mother_dict['all_z'][i][1] for i in range(L)], dim = 0).cpu()
    )

    mother_dict['z_mask_list'] = torch.cat(mother_dict['z_mask_list'], dim = 0).cpu()

    return mother_dict

def batch_forwards(model, X, times, batch_size = 64, org_v = False):
    '''
    Runs the model in batches for large datasets. Used to get lots of embeddings, outputs, etc.
        - Need to use this bc there's a specialized dictionary notation for output of the forward method (see concat_all_dicts)
    '''

    iters = torch.arange(0, X.shape[1], step = batch_size)
    out_list = []

    for i in range(len(iters)):
        if i == (len(iters) - 1):
            batch_X = X[:,iters[i]:,:]
            batch_times = times[:,iters[i]:]
        else:
            batch_X = X[:,iters[i]:iters[i+1],:]
            batch_times = times[:,iters[i]:iters[i+1]]

        with torch.no_grad():
            out = model(batch_X, batch_times, captum_input = False)

        out_list.append(out)

    out_full = concat_all_dicts(out_list, org_v = org_v)

    return out_full

def batch_forwards_TransformerMVTS(model, X, times, batch_size = 64):

    iters = torch.arange(0, X.shape[1], step = batch_size)
    out_list = []
    z_list = []

    for i in range(len(iters)):
        if i == (len(iters) - 1):
            batch_X = X[:,iters[i]:,:]
            batch_times = times[:,iters[i]:]
        else:
            batch_X = X[:,iters[i]:iters[i+1],:]
            batch_times = times[:,iters[i]:iters[i+1]]

        with torch.no_grad():
            out, z, _ = model(batch_X, batch_times, captum_input = False, get_agg_embed = True)

        out_list.append(out)
        z_list.append(z)

    ztotal = torch.cat(z_list, dim = 0)
    outtotal = torch.cat(out_list, dim = 0)

    return outtotal, ztotal

