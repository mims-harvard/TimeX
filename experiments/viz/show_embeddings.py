import torch
import torch.nn.functional as F
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations

# Models:
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import TimeXModel

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

# Plotting tools:
import matplotlib.pyplot as plt
from umap import UMAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def main(model, test, args):
    need_ptypes = False
    if isinstance(model, TimeXModel):
        if model.ablation_parameters.ptype_assimilation:
            need_ptypes = True

    X, times, y = test

    inds = torch.randperm(X.shape[1])[:1000]
    X = X[:,inds,:]
    times = times[:,inds]
    y = y[inds]

    # Load test embeddings:
    # Run through in batches:

    B = X.shape[1]
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


    out = torch.cat([o['z_mask_list'] for o in out_list], dim = 0)

    #z_test_org = out['z_mask_list']
    #z_test = z_test_org.transpose(1, 2).flatten(0, 1) # Shape (B, d_z, ne) -> (B x ne, d_z)
    z_test = F.normalize(out, dim = -1)
    z_test_np = z_test.detach().cpu().numpy()

    if need_ptypes:
        # Get prototypes too:
        ptype_z_np = F.normalize(model.prototypes, dim = -1).detach().cpu().numpy()
        to_fit_z = np.concatenate([z_test_np, ptype_z_np], axis = 0)
    else:
        to_fit_z = z_test_np

    m = UMAP(metric = 'cosine')

    # Fit UMAP reducer:
    m.fit(to_fit_z)

    # Start plotting: ---------------------
    plt.figure(dpi=200)
    
    # Now plot explanations - stratify by class:
    #y_np = y.detach().cpu().numpy()
    for yi in y.unique():
        yitem = yi.item()
        #zt_i = z_test_org[y == yi,:,:].transpose(1,2).flatten(0,1).detach().cpu().numpy()
        zt_i = z_test[y==yi,:].detach().cpu().numpy()

        zt_umap = m.transform(zt_i)

        plt.scatter(zt_umap[:,0], zt_umap[:,1], label = 'Class {:d}'.format(yitem), alpha = 0.5)

    plt.legend()

    # Show prototypes if needed
    if need_ptypes:
        pz_umap = m.transform(ptype_z_np)
        plt.scatter(pz_umap[:,0], pz_umap[:,1], alpha = 1.0, c = np.arange(pz_umap.shape[0]), cmap = 'viridis', marker = 's')

    if args.savepath is not None:
        Xnp = X.cpu().numpy()
        ynp = y.cpu().numpy()
        torch.save((Xnp, ynp, z_test_np, ptype_z_np), args.savepath)

    plt.show()

def eval_model(model, test):

    f1, out = eval_mv4(test, model, masked = False)
    print('Test F1 (unmasked): {:.4f}'.format(f1))

    f1, _ = eval_mv4(test, model, masked = True)
    print('Test F1 (masked): {:.4f}'.format(f1))

    sim = cosine_sim(out, test)
    print('Test cosine sim: {:.4f}'.format(sim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', type=int, default = 1)
    parser.add_argument('--org_v', action = 'store_true')
    parser.add_argument('--savepath', type = str, default = None)

    args = parser.parse_args()

    D = args.dataset.lower()

    if D == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
        test = D['test']
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
    elif D == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
        test = D['test']
    elif D == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
        test = D['test']
    elif D == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
        test = D['test']
    elif D == 'scs_fixone':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleFixOne')
        test = D['test']
    elif D == 'epilepsy':
        _, _, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
        test = (test.X, test.time, test.y)
    elif D == 'mitecg_simple':
        _, _, test = process_MITECG(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Simple/')
        test = (test.X, test.time, test.y)
    elif D == 'lowvardetect':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/LowVarDetect')
        test = D['test']
    elif D == 'mitecg_hard':
        trainD, _, test, _ = process_MITECG(split_no = args.split_no, device = device, hard_split = True, exclude_pac_pvc = True, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Hard/')
        test = (test.X, test.time, test.y)

    # Loading:
    print('Loading model at {}'.format(args.model_path))
    sdict, config = torch.load(args.model_path)
    print('Config:\n', config)

    # Prototype:
    if args.org_v:
        model = Modelv6_v2(**config)
    else:
        model = TimeXModel(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    #eval_model(model, test)

    main(model, test, args)