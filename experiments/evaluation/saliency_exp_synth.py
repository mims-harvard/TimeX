import re
import argparse
from pathlib import Path
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
from txai.models.bc_model import BCExplainModel

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

    elif args.dataset == 'seqcomb_mv':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 2,
            n_classes = 4,
            trans_dim_feedforward = 128,
            trans_dropout = 0.25,
            d_pe = 16,
        )

    #model = torch.compile(model)

    return model

def main(args):

    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShape')
    elif Dname == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingle')
    elif Dname == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingleBetter')
    elif Dname == 'seqcomb_mv':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombMV')
    elif Dname == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
    elif Dname == 'scs_inline':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleInline')
    elif Dname == 'scs_fixone':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleFixOne')
    
    test = D['test']


    if Dname == 'scs_better' or Dname == 'seqcombsingle' or Dname == 'scs_inline' or Dname == 'seqcomb_mv':
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
                    generated_exps[:,iters[i]:,:] = torch.stack(out['mask_in'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
                else:
                    generated_exps[:,iters[i]:iters[i+1],:] = torch.stack(out['mask_in'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
            else:
                if i == (len(iters) - 1):
                    if batch_X.shape[-1] == 1:
                        generated_exps[:,iters[i]:,:] = out['mask_in']
                    else:
                        generated_exps[:,iters[i]:,:] = out['mask_in'].transpose(0,1)
                else:
                    if batch_X.shape[-1] == 1:
                        generated_exps[:,iters[i]:iters[i+1],:] = out['mask_in']
                    else:
                        generated_exps[:,iters[i]:iters[i+1],:] = out['mask_in'].transpose(0,1)

    elif args.exp_method == "winit":
        from winit_wrapper import WinITWrapper, aggregate_scores # Moved here bc of import issues on Owen's side
        model = get_model(args, X)
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)
        model.eval()
        winit_path = Path(args.model_path).parent / f"winit_split={args.split_no}/"
        winit = WinITWrapper(
            device, 
            num_features=D["test"][0].shape[-1], 
            data_name=Dname, 
            path=winit_path
        )
        winit.set_model(model)
        winit.load_generators()
        # winit wrapper expects shape of (batch, num_features, num_times) for X
        # and (batch, num_times) for times
        X_perm = X.permute(1, 2, 0)
        times_perm = times.permute(1, 0)
        attribution = winit.attribute(X_perm, times_perm)
        # paper notes best performance with mean aggregation
        generated_exps = torch.from_numpy(aggregate_scores(attribution, "mean"))
        # permute (batch, features, times) back to (times, batch, features)
        generated_exps = generated_exps.permute(2, 0, 1)
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

    return results_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_method', type = str, help = "Options: ['ig', 'dyna', 'winit', 'ours']")
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1, type=int)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--org_v', action = 'store_true')
    parser.add_argument('--data_path', default="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/", type = str, help = 'path to datasets root')

    args = parser.parse_args()
    if args.split_no == -1:
        # eval results on all splits
        results = {}
        for split in range(1, 6):
            # replace model path with correct split
            args.model_path = re.sub("split=\d", f"split={split}", args.model_path)
            print("model path:", args.model_path)
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