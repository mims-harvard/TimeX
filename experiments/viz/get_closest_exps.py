import torch
import torch.nn.functional as F
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations

# Models:
from txai.models.modelv6 import Modelv6
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.modelv6_v2_concepts import Modelv6_v2_concepts
from txai.models.modelv6_v2_ptnew import Modelv6_v2_PT
from txai.models.modelv6_v3 import Modelv6_v3

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG
from txai.vis.visualize_mv6 import vis_exps_w_sim
from txai.vis.vis_saliency import vis_one_saliency_univariate
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.experimental import get_explainer

# Plotting tools:
import matplotlib.pyplot as plt
from umap import UMAP

from txai.prototypes.posthoc import find_kmeans_ptypes, find_nearest_explanations
from txai.models.run_model_utils import batch_forwards, batch_forwards_TransformerMVTS

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

    model.eval()

    return model

@torch.no_grad()
def main_clusters(model, train, test, args):

    Xtrain, times_train, y_train = train
    out_train = batch_forwards(model, Xtrain, times_train, batch_size = 64)

    Xtest, times_test, y_test = test
    out_test = batch_forwards(model, Xtest, times_test, batch_size = 64)

    print('zm test', out_test['z_mask_list'].shape)
    print('zm train', out_train['z_mask_list'].shape)

    cluster_train, cluster_test = find_kmeans_ptypes(out_train['z_mask_list'], out_test['z_mask_list'])

    if args.save_path is not None:
        torch.save((cluster_train, cluster_test), args.save_path)

    if args.show:
        # Get model outputs:
        Xtest, times_test, y_test = test
        out = model(Xtest, times_test, captum_input = False)

        z_test_org = out['z_mask_list']
        z_test = z_test_org.squeeze() # Squeeze out last dim
        z_test_np = z_test.detach().cpu().numpy()

        # Fit UMAP:
        m = UMAP()
        m.fit(z_test_np)

        # Start plotting: ---------------------
        plt.figure(dpi=200)
        
        # Now plot explanations - stratify by class:
        #y_np = y.detach().cpu().numpy()
        for c in np.unique(cluster_test):
            zt_i = z_test_np[c == cluster_test,:]
            zt_umap = m.transform(zt_i)
            plt.scatter(zt_umap[:,0], zt_umap[:,1], label = 'Cluster = {}'.format(c), alpha = 0.5)

        plt.legend()
        plt.show()

@torch.no_grad()
def main_sim(model, train, test, args):
    Xtrain, times_train, y_train = train
    out_train = batch_forwards(model, Xtrain, times_train, batch_size = 64)

    Xtest, times_test, y_test = test
    out_test = batch_forwards(model, Xtest, times_test, batch_size = 64)

    ztrain = out_train['z_mask_list'].squeeze(-1)
    ztest = out_test['z_mask_list'].squeeze(-1)

    # Choose random exps:
    if args.class_num is not None:
        to_choose = (y_test == args.class_num).nonzero(as_tuple=True)[0].cpu()
    else:
        to_choose = torch.arange(Xtest.shape[0])

    q_inds = to_choose[torch.randperm(to_choose.shape[0])[:3]]

    zq = ztest[q_inds,:]

    best_per_q = find_nearest_explanations(zq, ztrain, n_exps_per_q = args.nclose)

    print('best_per_q', best_per_q)

    # Index:
    Xq = Xtest[:,q_inds,:].detach().clone().cpu().numpy()
    mq = out_test['mask_logits'][q_inds,:,0].detach().cpu().numpy()

    mtrain = out_train['mask_logits'][:,:,0].detach().cpu().numpy()

    # Build lists:
    Xn_list, mn_list = [], []
    for i in range(best_per_q.shape[0]):
        ind = best_per_q[i,:]
        Xn_list.append(Xtrain[:,ind,:].detach().clone().cpu().numpy())
        mn_list.append(mtrain[ind,:])

    vis_exps_w_sim(X_query = Xq, mask_query = mq, 
        X_nearby_list = Xn_list, mask_nearby_list = mn_list,
        show = True)

def main_sim_other_exp(args, train, test):
    
    Xtrain, times_train, y_train = train
    Xtest, times_test, y_test = test

    model = get_model(args, Xtrain)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Get all train embeddings: - TransformerMVTS
    out_train, ztrain = batch_forwards_TransformerMVTS(model, Xtrain, times_train, batch_size = 64)

    # Choose random exps:
    if args.class_num is not None:
        to_choose = (y_test == args.class_num).nonzero(as_tuple=True)[0].cpu()
    else:
        to_choose = torch.arange(Xtest.shape[0])

    q_inds = to_choose[torch.randperm(to_choose.shape[0])[:3]]

    # Get embeddings of sampled:
    out_test, ztest, _ = model(Xtest[:,q_inds,:], times_test[:,q_inds], captum_input = False, get_agg_embed = True)

    # Compute sims:
    ztrain, ztest = F.normalize(ztrain, dim = 1), F.normalize(ztest, dim = 1)

    sims = torch.matmul(ztest, ztrain.transpose(0,1))
    best_sim_inds = sims.argsort(descending=True, dim = -1)[:,:args.nclose]

    # Get explainer:
    explainer, _ = get_explainer(key = args.exp_method, args = args, device = device)

    # Explain samples chosen:
    all_exps = []
    for i in range(3): # Over chosen test samples
        exp_i = []
        for j in range(args.nclose):
            ind = best_sim_inds[i,j]
            Xs = Xtrain[:,ind,:].clone()
            ts = times_train[:,ind].clone().unsqueeze(1)
            ys = y_train[ind].unsqueeze(0).clone()
            if args.exp_method == 'dyna':
                exp = explainer(model, Xs, ts, y = ys)
            else:
                exp = explainer(model, Xs.unsqueeze(1), ts, y = ys)
            exp_i.append(exp)

        all_exps.append(exp_i)

    # Explain test samples:
    all_test_exps = []
    for j in range(3):
        i = q_inds[j]
        Xs = Xtest[:,i,:].clone()
        ts = times_test[:,i].clone().unsqueeze(1)
        ys = y_test[i].clone()
        if args.exp_method == 'dyna':
            exp = explainer(model, Xs, ts, y = ys.unsqueeze(0))
        else:
            exp = explainer(model, Xs.unsqueeze(1), ts, y = ys)
        all_test_exps.append(exp)

    # Visualize:
    fig, ax = plt.subplots(args.nclose + 1, 3, dpi = 200, sharex = True)

    for i in range(3):
        
        ti = q_inds[i]
        vis_one_saliency_univariate(Xtest[:,ti,:], all_test_exps[i][:,:,0], ax[0,i], fig)

        for j in range(args.nclose):
            ind = best_sim_inds[i,j]
            vis_one_saliency_univariate(Xtrain[:,ind,:], all_exps[i][j][:,:,0], ax[(j+1),i], fig)
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_method', type = str, default = 'ours')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', type=int, default = 1)
    parser.add_argument('--save_path', type = str, default = None)
    parser.add_argument('--class_num', type = int, default = None)
    parser.add_argument('--nclose', type = int, default = 4)

    args = parser.parse_args()

    D = args.dataset.lower()

    if D == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        test = D['test']
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
    elif D == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        test = D['test']
    elif D == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        test = D['test']
    elif D == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        test = D['test']
    elif D == 'scs_fixone':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleFixOne')
        train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        test = D['test']
    elif D == 'epilepsy':
        trainD, _, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
        train = (trainD.X, trainD.time, trainD.y)
        test = (test.X, test.time, test.y)
    elif D == 'mitecg_simple':
        trainD, _, test = process_MITECG(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Simple/')
        train = (trainD.X, trainD.time, trainD.y)
        test = (test.X, test.time, test.y)

    # Loading:
    print('Loading model at {}'.format(args.model_path))

    # Prototype:
    if args.exp_method == 'ours':
        sdict, config = torch.load(args.model_path)
        print('Config:\n', config)
        model = Modelv6_v2(**config)
        model.load_state_dict(sdict)
        model.eval()
        model.to(device)

        main_sim(model, train, test, args)
    else:
        main_sim_other_exp(args, train, test)