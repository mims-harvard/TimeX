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

from txai.models.eliminates.modelv6_v2_for_idexp import Modelv6_v2

from txai.utils.evaluation import ground_truth_xai_eval
from visualize_mv6 import visualize_explanations

def make_external_identifier(train, val, test):
    train_X = train[0].clone()
    train_times = train[1].clone()
    train_y = train[2].clone()
    train_inds = torch.arange(train_X.shape[1])

    val_X = val[0].clone()
    val_times = val[1].clone()
    val_y = val[2].clone()
    val_inds = torch.arange(val_X.shape[1]) + 1 + train_inds.max()

    test_X = test[0].clone()
    test_times = test[1].clone()
    test_y = test[2].clone()
    test_inds = torch.arange(test_X.shape[1]) + 1 + val_inds.max()

    whole_X = torch.cat([train_X, val_X, test_X], dim = 1)
    
    # Find identifiers for the whole X pattern:
    comb_inds = torch.combinations(torch.arange(whole_X.shape[0]), r = 2, with_replacement = False)

    for i in range(whole_X.shape[1]):
        j, k = comb_inds[i,0], comb_inds[i,1]
        whole_X[j:(j+10), i, 0] = 5.0
        whole_X[k:(k+10), i, 0] = 5.0

    # Sample into inds:
    external_train = (whole_X[:,train_inds,:], train_times, train_y)
    external_val = (whole_X[:,val_inds,:], val_times, val_y)
    external_test = (whole_X[:,test_inds,:], test_times, test_y)

    return external_train, external_val, external_test

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


    D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    
    test = D['test']
    train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
    val = D['val']

    _, _, test = make_external_identifier(train, val, test)


    y = test[2]
    X = test[0][:,(y != 0),:]
    times = test[1][:,y != 0]
    gt_exps = D['gt_exps'][:,(y != 0).detach().cpu(),:]
    y = y[y != 0]
    T, B, d = X.shape

    if args.exp_method == 'ours':
        sdict, config = torch.load(args.model_path)
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
                #print(torch.stack(out['mask_in'], dim = 0).sum(dim=0).shape)
                generated_exps[:,iters[i]:,:] = torch.stack(out['mask_in'], dim = 0).sum(dim=0).unsqueeze(-1).transpose(0,1)
            else:
                # print(out['mask_in'][0].shape)
                # print(torch.stack(out['mask_in'], dim = 0).sum(dim=0).shape)
                # exit()
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
            generated_exps[:,i,:] = exp
    
    
    results_dict = ground_truth_xai_eval(generated_exps, gt_exps)

    # Show all results:
    print('Results for {} explainer on {} with split={}'.format(args.exp_method, args.dataset, args.split_no))
    for k, v in results_dict.items():
        print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))

    # Visualize:
    visualize_explanations(model, test, test_X_regular = D['test'][0], class_num = 0)
    visualize_explanations(model, test, test_X_regular = D['test'][0], class_num = 1)
    visualize_explanations(model, test, test_X_regular = D['test'][0], class_num = 2)
    visualize_explanations(model, test, test_X_regular = D['test'][0], class_num = 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_method', type = str, help = "Options: ['ig', 'dyna', 'ours']", default = 'ours')
    parser.add_argument('--dataset', type = str, default = 'scs_better')
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--model_path', type = str, default = 'models/ours_split=1.pt')

    args = parser.parse_args()

    main(args)