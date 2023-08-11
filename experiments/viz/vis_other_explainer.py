import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.experimental import get_explainer
from txai.vis.vis_saliency import vis_one_saliency
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, X):

    if args.dataset == 'scs_better':
        if args.arch == 'cnn':
            model = CNN(
                d_inp = X.shape[-1],
                n_classes = 4,
            )
        elif args.arch == 'lstm':
            model = LSTM(
                d_inp = X.shape[-1],
                n_classes = 4,
            )
        else:
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
        if args.arch == 'cnn':
            model = CNN(
                d_inp = X.shape[-1],
                n_classes = 4,
            )
        elif args.arch == 'lstm':
            model = LSTM(
                d_inp = X.shape[-1],
                n_classes = 4,
            )
        else:
            model = TransformerMVTS(
                d_inp = X.shape[-1],
                max_len = X.shape[0],
                n_classes = 4,
                trans_dim_feedforward = 16,
                trans_dropout = 0.1,
                d_pe = 16,
            )
    
    elif args.dataset == 'epilepsy':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
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
    
    elif args.dataset == 'mitecg_simple':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 1,
            n_classes = 2,
            trans_dim_feedforward = 32,
            trans_dropout = 0.1,
            d_pe = 16,
        )
    elif args.dataset == 'lowvardetect':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 1,
            n_classes = 4,
            trans_dim_feedforward = 32,
            trans_dropout = 0.25,
            d_pe = 16,
            stronger_clf_head = False,
        )
    elif args.dataset == 'mitecg_hard':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
            nlayers = 1,
            nhead = 1,
            trans_dim_feedforward = 64,
            trans_dropout = 0.1,
            d_pe = 16,
            stronger_clf_head = False,
            pre_agg_transform = False,
            norm_embedding = True
        )

    return model

def main(test, args):

    X, time, y = test
    T, B, d = X.shape

    # Load model:
    model = get_model(args, X)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Sample 3:
    choices = np.arange(B)
    if args.class_num is not None:
        ynp = y.detach().clone().cpu().numpy()
        choices = choices[ynp == args.class_num]

    np.random.seed(args.sample_seed)
    inds = np.random.choice(choices, size = (3,), replace = False)
    sampX, samptime, sampy = X[:,inds,:], time[:,inds], y[inds]
    sampX = sampX.to(device)
    samptime = samptime.to(device)
    sampy = sampy.to(device)

    if args.exp_method == 'gt':
        generated_exps = gt_exps[:,inds,:]
    else:
        explainer, _ = get_explainer(key = args.exp_method, args = args, device = device)
        generated_exps = torch.zeros_like(sampX)
        for i in range(3):
            if args.exp_method == 'dyna': # This is a lazy solution, fix later
                print('x', sampX[:,i,:].shape)
                print('t', samptime[:,i].unsqueeze(1).shape)
                exp = explainer(model, sampX[:,i,:].clone(), samptime[:,i].clone().unsqueeze(1), y = sampy[i].unsqueeze(0).clone())
            else:
                exp = explainer(model, sampX[:,i,:].unsqueeze(1).clone(), samptime[:,i].unsqueeze(-1).clone(), sampy[i].unsqueeze(0).clone())
            generated_exps[:,i,:] = exp

    # Prediction pass through model:
    out = model(sampX, samptime, captum_input = False)
    pred = out.softmax(dim=-1).argmax(dim=-1)

    fig, ax = plt.subplots(d, 3, sharex = True, squeeze = False)

    #ax[0,0].set_title('test')

    for i in range(3):
        vis_one_saliency(sampX[:,i,:], generated_exps[:,i,:], ax, fig, col_num = i)
        ax[0,i].set_title('y = {:d}, yhat = {:d}'.format(sampy[i].item(), pred[i].item()))
    
    #fig.set_size_inches(18.5, 3 * d)
    fig.set_size_inches(18, 5)
    if args.savepdf is not None:
        plt.savefig(args.savepdf)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_method', type = str, help = "Options: ['ig']")
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--class_num', default = None, type = int)
    parser.add_argument('--sample_seed', default = None, type = int)
    parser.add_argument('--savepdf', default = None)
    parser.add_argument('--arch', default = None)

    args = parser.parse_args()

    D = args.dataset.lower()

    # Switch on loading test data:
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
        gt_exps = D['gt_exps']
    elif D == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
        test = D['test']
    elif D == 'scs_inline':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleInline')
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
        _, _, test, _ = process_MITECG(split_no = args.split_no, hard_split = True, need_binarize = True, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Hard/')
        test = (test.X, test.time, test.y)

    main(test, args)