import torch
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations

# Models:
from txai.models.modelv6 import Modelv6
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.modelv6_v2_concepts import Modelv6_v2_concepts
from txai.models.modelv6_v3 import Modelv6_v3

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

from txai.utils.functional import cosine_sim_matrix

# Plotting tools:
import matplotlib.pyplot as plt
from umap import UMAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model, test, args):

    X, times, y = test

    # Get concept matches for each testing sample passed through model:
    out = model(X, times, captum_input = False)

    z_test_e = out['z_mask_list'] # Shape (B, d_z, n_e)
    Ne = z_test_e.shape[-1]

    z_concepts = model.get_concept_embeddings(args.n_aug) # Shape (n_c, n_aug, d_z)

    sim_mats = []

    for yi in y.unique():
        ze_sub = z_test_e[y == yi,:,:]
        print('Label {:d}'.format(yi.item()))
        for i in range(z_concepts.shape[0]):
            for j in range(Ne):
                sim_mat = cosine_sim_matrix(ze_sub[:,:,j], z_concepts[i,:,:])

                mean_dist = sim_mat.mean()
                stderr_dist = sim_mat.std() / np.sqrt(sim_mat.flatten().shape[0])

                print('\tC = {}, E = {} \t sim = {:.4f} +- {:.4f}'.format(i, j, mean_dist.item(), stderr_dist.item()))
        print(' ---- ')


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
    parser.add_argument('--n_aug', type = int, default = 100, help = 'Number of augmentations per concept')
    parser.add_argument('--c_embed_path', type = str, default = None, help = 'Path to embeddings of concepts, not required')
    parser.add_argument('--x_embed_path', type = str, default = None, help = 'Path to embeddings of test samples to plot')

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

    model = Modelv6_v2_concepts(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    # Evaluating:
    eval_model(model, test)
    
    # Call main visualization:
    main(model, test, args)