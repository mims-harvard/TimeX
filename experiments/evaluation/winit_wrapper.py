from argparse import ArgumentParser
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from txai.utils.data import process_Synth
from txai.baselines.WinIT.winit.explainer.winitexplainers import WinITExplainer
from txai.synth_data.simple_spike import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_generator(args):
    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShape')
    elif Dname == 'seqcombsingle':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingle')
    elif Dname == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingleBetter')
    elif Dname == 'freqshapeud':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShapeUD')
    
    winit_path = Path(args.models_path) / f"winit_split={args.split_no}/"

    winit = WinITExplainer(
        device, 
        num_features=D["test"][0].shape[-1], 
        data_name=Dname, 
        path=winit_path
    )

    # NOTE: WinIT code expects time series of shape [n, features, time]
    train_input = torch.stack([D["train_loader"][i][0].permute(1, 0) for i in range(len(D["train_loader"]))])
    train_label = torch.stack([D["train_loader"][i][2] for i in range(len(D["train_loader"]))])
    train_ds = TensorDataset(train_input, train_label)
    # time, n, features -> n, features, time
    val_ds = TensorDataset(D["val"][0].permute(1, 2, 0), D["val"][2])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    results = winit.train_generators(train_loader=train_dl, valid_loader=val_dl)

    plt.plot(results.train_loss_trends[0], label="train_loss")
    plt.plot(results.valid_loss_trends[0], label="valid_loss")
    plt.axvline(results.best_epochs[0], label="best_epoch", ls="--", color="black")
    plt.legend()
    plt.savefig(winit_path / "loss.png")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1)
    parser.add_argument('--models_path', type = str, help = 'path to store models')
    parser.add_argument('--data_path', default="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/", type = str, help = 'path to datasets root')
    args = parser.parse_args()
    train_generator(args)