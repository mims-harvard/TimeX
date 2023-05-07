import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.experimental import get_explainer
from txai.vis.vis_saliency import vis_one_saliency
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset

from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.modelv6_v2_concepts import Modelv6_v2_concepts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, X):

    if args.dataset == 'scs_better':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 4,
            nlayers = 2,
            nhead = 1,
            trans_dim_feedforward = 64,
            trans_dropout = 0.25,
            d_pe = 16,
        )

    elif args.dataset == 'freqshape':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 4,
            trans_dim_feedforward = 16,
            trans_dropout = 0.1,
            d_pe = 16,
        )

    #model = torch.compile(model)

    return model

'''
    (B,nc)      'pred': pred_regular, # Prediction on regular embedding (prediction branch)
    (B,nc)      'pred_mask': pred_mask, # Prediction on masked embedding
    (B, T, ne)  'mask_logits': torch.stack(mask_in_list, dim = -1), # Mask logits, i.e. before reparameterization + ste
    (B, 1, ne)  'concept_scores': score_tensor,
    (B, T, ne)  'ste_mask': torch.stack(ste_mask_list, dim=-1),
    [(T, B, d)] 'smooth_src': smooth_src_list,
    (B, ne)     'p': torch.cat(p_list, dim=-1),
    ( (B, d) )  'all_z': (z_main, agg_z_c),
    (B, d, ne)  'z_mask_list': torch.stack(z_mask_list, dim = -1),
    []          'concept_selections_inds': cs_inds_list - 
'''

def concat_all_dicts(dlist):
    # Marries together all dictionaries
    # Will change based on output from model

    # print('dlist', dlist[0])
    # exit()

    mother_dict = {k:[] for k in dlist[0].keys()}

    is_tensor_list = []

    for d in dlist:
        for k in d.keys():
            if k == 'smooth_src':
                mother_dict[k].append(torch.stack(d[k], dim = -1))
            else:   
                mother_dict[k].append(d[k])


    mother_dict['pred'] = torch.cat(mother_dict['pred'], dim = 0).cpu()
    mother_dict['pred_mask'] = torch.cat(mother_dict['pred_mask'], dim = 0).cpu()
    mother_dict['mask_logits'] = torch.cat(mother_dict['mask_logits'], dim = 0).cpu()
    mother_dict['concept_scores'] = torch.cat(mother_dict['concept_scores'], dim = 0).cpu()
    mother_dict['ste_mask'] = torch.cat(mother_dict['ste_mask'], dim = 0).cpu()
    # [[(), ()], ... 24]
    mother_dict['smooth_src'] = torch.cat(mother_dict['smooth_src'], dim = 1).cpu() # Will be (T, B, d, ne)
    mother_dict['p'] = torch.cat(mother_dict['p'], dim = 0).cpu()

    L = len(mother_dict['all_z'])
    mother_dict['all_z'] = (
        torch.cat([mother_dict['all_z'][i][0] for i in range(L)], dim = 0).cpu(), 
        torch.cat([mother_dict['all_z'][i][1] for i in range(L)], dim = 0).cpu()
    )

    mother_dict['z_mask_list'] = torch.cat(mother_dict['z_mask_list'], dim = 0).cpu()

    mother_dict['concept_selections_inds'] = []

    return mother_dict
    

def main(args):

    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
    elif Dname == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    elif Dname == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    elif Dname == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
    
    test = D['test']


    if Dname == 'scs_better' or Dname == 'seqcombsingle':
        y = test[2]
        X = test[0][:,(y != 0),:]
        times = test[1][:,y != 0]
        gt_exps = D['gt_exps'][:,(y != 0).detach().cpu(),:]
        y = y[y != 0]
    else:
        X, times, y = test
        gt_exps = D['gt_exps']
    T, B, d = X.shape

    sdict, config = torch.load(args.model_path)
    if args.concept_v:
        model = Modelv6_v2_concepts(**config)
    else:
        model = Modelv6_v2(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    # Keep batch size at 64:
    iters = torch.arange(0, B, step = 64)
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

        # for k, v in out.items():
        #     print('{}: {}'.format(k, type(v)))
        #     if k == 'concept_selections_inds':
        #         continue
        #     if isinstance(v, torch.Tensor):
        #         print('Tensor shape', v.shape)
        #     else:
        #         print('Inner shape:', v[0].shape)
        #     print('-' * 50)

        out_list.append(out)


    out_main = concat_all_dicts(out_list)

    for k, v in out_main.items():
        print('{}: {}'.format(k, type(v)))
        if k == 'concept_selections_inds':
            break
        if isinstance(v, torch.Tensor):
            print('Tensor shape', v.shape)
        else:
            print('Inner shape:', v[0].shape)
        print('-' * 50)
        
    # Save to save_path:
    if args.only_outputs or (args.concept_v):
        torch.save(out_main, args.save_path)
    else:
        concept_embeddings = model.concept_dists.detach().clone().cpu()
        torch.save((out_main, concept_embeddings), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--save_path', type = str)
    parser.add_argument('--only_outputs', action = 'store_true')
    parser.add_argument('--concept_v', action = 'store_true')

    args = parser.parse_args()

    main(args)
