import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds

from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import BCExplainModel

from txai.prototypes.tune_ptypes import tune_ptypes
from txai.utils.predictors.loss_cl import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sim_criterion = EmbedConsistencyLoss()

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
       
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    sdict, config = torch.load(args.model_path) 
    model = BCExplainModel(**config)
    model.load_state_dict(sdict)

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze prototypes:
    model.prototypes.requires_grad = True

    optimizer = torch.optim.Adam([model.prototypes], lr = 1e-3, weight_decay = 0.0)

    tune_ptypes(model, optimizer, train_loader, num_epochs = 50, sim_criterion = sim_criterion)

    new_name = args.model_path[:-3] + '_ptune.pt'
    print(f'saving at {new_name}')

    model.save_state(new_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--org_v', action = 'store_true')
    parser.add_argument('--data_path', default="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/", type = str, help = 'path to datasets root')

    args = parser.parse_args()

    main(args)