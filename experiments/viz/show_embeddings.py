import torch
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

# Plotting tools:
import matplotlib.pyplot as plt
from umap import UMAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model, test, args):

    X, times, y = test

    inds = torch.randperm(X.shape[1])[:1000]
    X = X[:,inds,:]
    times = times[:,inds]
    y = y[inds]

    # Load test embeddings:
    out = model(X, times)

    z_test_org = out['z_mask_list']
    z_test = z_test_org.transpose(1, 2).flatten(0, 1) # Shape (B, d_z, ne) -> (B x ne, d_z)
    z_test_np = z_test.detach().cpu().numpy()

    m = UMAP()

    # Fit UMAP reducer:
    m.fit(z_test_np)

    # Start plotting: ---------------------
    plt.figure(dpi=200)
    
    # Now plot explanations - stratify by class:
    #y_np = y.detach().cpu().numpy()
    for yi in y.unique():
        yitem = yi.item()
        zt_i = z_test_org[y == yi,:,:].transpose(1,2).flatten(0,1).detach().cpu().numpy()

        zt_umap = m.transform(zt_i)

        plt.scatter(zt_umap[:,0], zt_umap[:,1], label = 'Class {:d}'.format(yitem), alpha = 0.5)

    plt.legend()
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

    # Loading:
    print('Loading model at {}'.format(args.model_path))
    sdict, config = torch.load(args.model_path)
    print('Config:\n', config)

    # Prototype:
    model = Modelv6_v2(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    eval_model(model, test)

    main(model, test, args)