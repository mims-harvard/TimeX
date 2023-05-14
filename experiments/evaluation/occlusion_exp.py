import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.experimental import get_explainer
from txai.vis.vis_saliency import vis_one_saliency
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset

from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import BCExplainModel

from txai.utils.functional import transform_to_attn_mask
from txai.utils.data.preprocess import process_Epilepsy, process_PAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, X):

    if args.dataset == 'epilepsy':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
            nlayers = 1,
            trans_dim_feedforward = 16,
            trans_dropout = 0.1,
            d_pe = 16,
            norm_embedding = False,
        )

    elif args.dataset == 'pam':
        model = TransformerMVTS(
            d_inp = X.shape[2],
            max_len = X.shape[0],
            n_classes = 8,
        )

    elif args.dataset == 'boiler':
        pass

    #model = torch.compile(model)

    return model

def main(args):

    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == 'epilepsy':
        trainEpi, val, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
        test = (test.X, test.time, test.y)
        trainX = trainEpi.X
    elif Dname == 'pam':
        trainPAM, val, test = process_PAM(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/PAMAP2data/', gethalf = True)
        test = (test.X, test.time, test.y)
        trainX = trainPAM.X
    elif Dname == 'boiler':
        pass
    else:
        raise ValueError('{} is not a valid dataset for stability'.format(Dname))

    X, times, y = test

    T, B, d = X.shape

    if args.exp_method == 'ours':
        sdict, config = torch.load(args.model_path)
        print('Config', config)
        if args.org_v:
            model = Modelv6_v2(**config)
        else:
            model = BCExplainModel(**config)
        model.load_state_dict(sdict)
        model.eval()
        model.to(device)

        # Keep batch size at 64:
        iters = torch.arange(0, B, step = 64)
        generated_exps = torch.zeros_like(X)

        for i in range(len(iters)):
            if i == (len(iters) - 1):
                batch_X = X[:,iters[i]:,:]
                batch_times = times[:,iters[i]:]
            else:
                batch_X = X[:,iters[i]:iters[i+1],:]
                batch_times = times[:,iters[i]:iters[i+1]]

            with torch.no_grad():
                out = model.get_saliency_explanation(batch_X, batch_times, captum_input = False)


            # NOTE: below capability only works with univariate for now - will need to edit after adding MV to model
            if args.org_v:
                if i == (len(iters) - 1):
                    generated_exps[:,iters[i]:,:] = torch.stack(out['ste_mask'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
                else:
                    generated_exps[:,iters[i]:iters[i+1],:] = torch.stack(out['ste_mask'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
            else:
                if i == (len(iters) - 1):
                    if batch_X.shape[-1] == 1:
                        generated_exps[:,iters[i]:,:] = out['ste_mask']
                    else:
                        generated_exps[:,iters[i]:,:] = out['ste_mask'].transpose(0,1)

                else:
                    if batch_X.shape[-1] == 1:
                        generated_exps[:,iters[i]:iters[i+1],:] = out['ste_mask']
                    else:
                        generated_exps[:,iters[i]:iters[i+1],:] = out['ste_mask'].transpose(0,1)

    else: # Use other explainer APIs:
        model = get_model(args, X)
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)
        model.eval()

        explainer, needs_training = get_explainer(key = args.exp_method, args = args, device = device)

        generated_exps = torch.zeros_like(X)

        for i in trange(B):
            # Eval all explainers:
            if args.exp_method == 'dyna': # This is a lazy solution, fix later
                exp = explainer(model, X[:,i,:].clone(), times[:,i].clone().unsqueeze(1), y = y[i].unsqueeze(0).clone())
            else:
                exp = explainer(model, X[:,i,:].unsqueeze(1).clone(), times[:,i].unsqueeze(-1).clone(), y[i].unsqueeze(0).clone())
            #print(exp.shape)
            generated_exps[:,i,:] = exp

    # Forward pass:
    if args.exp_method == 'ours':
        # Make passes in batches:
        m = model.forward_pass_ge(src = X)
    else:
        # Perturb unimportant parts of the input:
        # Replace with baseline and attention mask:
        featwise_mu = trainX.mean(dim=1)
        featwise_std = trainX.std(dim=1)

        baseline = torch.stack([torch.normal(mean = featwise_mu, std = featwise_std) for _ in range(X.shape[1])], dim = 1).to(X.device)

        # Get perturbation mask:
        perturb_mask = torch.zeros_like(X).bool()
        for i in range(B):
            exp = generated_exps[:,i,:]
            thresh = torch.quantile(exp, args.lower_pct, interpolation='nearest')
            if thresh.isnan().any():
                print('thresh', thresh)
            perturb_mask[:,i,:] = (exp > thresh) # Masks in all values lower than the median (50th percentile)
            if args.upper_pct is not None:
                upperthresh = torch.quantile(exp, args.lower_pct, interpolation='nearest')
                perturb_mask[:,i,:] = (exp > thresh) & (exp < upperthresh)
        
        perturb_mask = perturb_mask.float().to(X.device)
        Xperturb = (X * perturb_mask + (1 - perturb_mask) * baseline)
        #print('x perturb', Xperturb.isnan().any())
        seq_mask = (perturb_mask.sum(dim=-1) > 0).transpose(0,1).to(Xperturb.device) # New size: (B, T)
        attn_mask = transform_to_attn_mask(seq_mask)
        # print('attn mask', attn_mask.isnan().any())
        # print('attn_mask', attn_mask)

        # See how perturb mask and seqmask are similar:
        mean_eq = (perturb_mask == seq_mask.transpose(0,1).unsqueeze(-1)).sum(dim=0).float().mean()
        #print(f'Mean eq {mean_eq}, len = {perturb_mask.shape[0]}')

        with torch.no_grad():
            pred = model(Xperturb, times, attn_mask = attn_mask.float()) 

    # Get evaluations:
    pred_prob = pred.softmax(dim=-1).detach().clone().cpu()
    yc = y.cpu().numpy()
    one_hot_y = np.zeros((yc.shape[0], yc.max() + 1))
    #print(yc)
    for i, yi in enumerate(yc):
        one_hot_y[i,yi] = 1
    #print(one_hot_y)
    auprc_val = average_precision_score(one_hot_y, pred_prob, average = 'macro')

    auroc_val = roc_auc_score(one_hot_y, pred_prob, average = 'macro', multi_class = 'ovo')

    # Show all results:
    print('Results for {} explainer on {} with split={}'.format(args.exp_method, args.dataset, args.split_no))
    print('AUPRC = {:.4f}'.format(auprc_val))
    print('AUROC = {:.4f}'.format(auroc_val))

    return {'auroc': auroc_val, 'auprc': auprc_val}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_method', type = str, help = "Options: ['ig', 'dyna', 'winit', 'ours']")
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1, type = int)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--org_v', action = 'store_true')
    parser.add_argument('--data_path', default="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/", type = str, help = 'path to datasets root')
    parser.add_argument('--lower_pct', default = 0.9, type = float)
    parser.add_argument('--upper_pct', default = None, type = float)

    args = parser.parse_args()

    perm_model_path = args.model_path

    if (args.split_no == -1):
        # eval results on all splits
        results = {}
        for split in range(1, 6):
            # TODO don't hard code
            args.model_path = perm_model_path.format(split)
            args.split_no = split
            split_results = main(args)
            for k, v in split_results.items():
                if k not in results:
                    results[k] = []
                results[k].extend(v)
        print('Results for {} explainer on all splits'.format(args.exp_method, args.dataset))
        for k, v in results.items():
            print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))
    else:
        main(args)