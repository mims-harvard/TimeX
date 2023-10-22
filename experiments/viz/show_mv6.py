import torch
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations, visualize_explanations_new

# Models:
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import TimeXModel

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model, test, class_num, heatmap = False, seed = None, topk = None, savepdf = None):

    visualize_explanations_new(model, test, show = False, class_num = class_num, heatmap = heatmap, topk = topk, seed = seed)
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(18, 5)
    if savepdf is not None:
        plt.savefig(savepdf)
    plt.show()

def show_embed(model, test, caption = None, save_embed_name = None):
    
    X, times, y = test

    model.eval()
    with torch.no_grad():
        out_dict = model(X, times, captum_input = False)

    # Deconstruct dict:
    full_z, mask_z = out_dict['all_z'] # Both should be (B, d)
    print('Full z', full_z.shape)
    print('Mask z', mask_z.shape)

    fz, mz = full_z.detach().cpu().numpy(), mask_z.detach().cpu().numpy()

    ids = np.concatenate([np.zeros(fz.shape[0]), np.ones(mz.shape[0])])
    joint_z = np.concatenate([fz, mz], axis=0)

    tsne_z = TSNE().fit_transform(joint_z)

    plt.scatter(tsne_z[:,0], tsne_z[:,1], c = ids, alpha = 0.5)
    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
    plt.show()

    if save_embed_name is not None:
        np.save('full_{}.pt'.format(save_embed_name), fz)
        np.save('mask_{}.pt'.format(save_embed_name), mz)

    sm = sim_mat(full_z, mask_z)

    plt.imshow(sm)
    cbar = plt.colorbar()
    cbar.set_label('Cosine Similarity')
    plt.show()

def eval_model(model, test):

    f1, out = eval_mv4(test, model, masked = False)
    print('Test F1 (unmasked): {:.4f}'.format(f1))

    f1, _ = eval_mv4(test, model, masked = True)
    print('Test F1 (masked): {:.4f}'.format(f1))

    # sim = cosine_sim(out, test)
    # print('Test cosine sim: {:.4f}'.format(sim))

def show_concepts(model, test):
    vis_concepts(model, test, show = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--embedding', action = 'store_true')
    parser.add_argument('--class_num', type=int, default = 0)
    parser.add_argument('--split_no', type=int, default = 1)
    parser.add_argument('--discrete', action='store_true', help = 'Shows mask as discrete object')
    parser.add_argument('--sample_seed', type = int)
    parser.add_argument('--save_emb_name', type = str)
    parser.add_argument('--topk', type = int, default = None)
    parser.add_argument('--show_concepts', action = 'store_true', help = 'shows discovered concepts, if applicable')
    parser.add_argument('--savepdf', type = str, default = None)

    parser.add_argument('--org_v', action = 'store_true')

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
    elif D == 'seqcomb_mv':
        D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombMV')
        test = D['test']
    elif D == 'mitecg_hard':
        _, _, test, _ = process_MITECG(split_no = args.split_no, hard_split = True, need_binarize = True, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Hard/')
        test = (test.X, test.time, test.y)

    # Loading:
    print('Loading model at {}'.format(args.model_path))
    sdict, config = torch.load(args.model_path)
    #print('Config:\n', config)
    #exit()
    if args.org_v:
        model = Modelv6_v2(**config)
    else:
        model = TimeXModel(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    # Evaluating:
    #eval_model(model, test)

    # Vis concepts:
    #vis_concepts(model, test)
    
    # Call main visualization:
    main(model, test, args.class_num, heatmap = (not args.discrete), seed = args.sample_seed, topk = args.topk, savepdf = args.savepdf)

    if args.show_concepts:
        show_concepts(model, test)

    # if args.embedding:
    #     #show_embed(model, train)
    #     show_embed(model, test, save_embed_name = args.save_emb_name)