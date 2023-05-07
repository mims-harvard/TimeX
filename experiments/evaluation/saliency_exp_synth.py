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

from txai.utils.evaluation import ground_truth_xai_eval

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
    
    elif args.dataset == 'scs_inline':
        model = TransformerMVTS(
            d_inp = 1,
            max_len = 200,
            n_classes = 4,
            nlayers = 2,
            nhead = 1,
            trans_dim_feedforward = 128,
            trans_dropout = 0.2,
            d_pe = 16,
            # aggreg = 'mean',
            # norm_embedding = True
        )
    
    elif args.dataset == 'scs_fixone':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 2,
            n_classes = 4,
            trans_dim_feedforward = 32,
            trans_dropout = 0.1,
            d_pe = 16,
        )

    #model = torch.compile(model)

    return model

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
    elif Dname == 'scs_inline':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleInline')
    elif Dname == 'scs_fixone':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleFixOne')
    
    test = D['test']


    if Dname == 'scs_better' or Dname == 'seqcombsingle' or Dname == 'scs_inline':
        y = test[2]
        X = test[0][:,(y != 0),:]
        times = test[1][:,y != 0]
        gt_exps = D['gt_exps'][:,(y != 0).detach().cpu(),:]
        y = y[y != 0]
    else:
        X, times, y = test
        gt_exps = D['gt_exps']
    T, B, d = X.shape

    if args.exp_method == 'ours':
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
            if i == (len(iters) - 1):
                generated_exps[:,iters[i]:,:] = torch.stack(out['mask_in'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
            else:
                generated_exps[:,iters[i]:iters[i+1],:] = torch.stack(out['mask_in'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)

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
    
    
    results_dict = ground_truth_xai_eval(generated_exps, gt_exps)

    # Show all results:
    print('Results for {} explainer on {} with split={}'.format(args.exp_method, args.dataset, args.split_no))
    for k, v in results_dict.items():
        print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_method', type = str, help = "Options: ['ig', 'dyna', 'ours']")
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--concept_v', action = 'store_true')

    args = parser.parse_args()

    main(args)