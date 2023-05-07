import torch
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations, vis_prototypes

# Models:
from txai.models.modelv6 import Modelv6
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.modelv6_v2_concepts import Modelv6_v2_concepts
#from txai.models.modelv6_v2_ptype import Modelv6_v2_ptype
from txai.models.modelv6_v2_ptnew import Modelv6_v2_PT
from txai.models.modelv6_v3 import Modelv6_v3

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(model, test):

    f1, out = eval_mv4(test, model, masked = False)
    print('Test F1 (unmasked): {:.4f}'.format(f1))

    f1, _ = eval_mv4(test, model, masked = True)
    print('Test F1 (masked): {:.4f}'.format(f1))

    # sim = cosine_sim(out, test)
    # print('Test cosine sim: {:.4f}'.format(sim))

def main(model, test):
    mpl.rcParams['figure.dpi'] = 200
    vis_prototypes(model, test, show = True, k = 5)

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
    model = Modelv6_v2_PT(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    # Evaluating:
    eval_model(model, test)

    # Vis concepts:
    #vis_concepts(model, test)
    
    # Call main visualization:
    main(model, test)

    # if args.embedding:
    #     #show_embed(model, train)
    #     show_embed(model, test, save_embed_name = args.save_emb_name)