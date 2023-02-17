import torch

from txai.vis.visualize_cbm1 import visualize
from txai.models.cbmv1 import CBMv1
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    D = process_Synth(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    model = CBMv1(
        d_inp = 1,
        max_len = 50,
        n_classes = 4,
        n_concepts = 2,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        norm_embedding = False,
        type_masktoken = 'norm_datawide',
        masktoken_kwargs = {'mu': mu, 'std': std},
        mask_seed = 1234,
    )
    model.to(device)

    spath = 'models/Scomb_cbm_split=1.pt'

    model.load_state_dict(torch.load(spath))
    model.load_concept_bank('concept_bank.pt')

    visualize(model, test, show = False, class_num = 3)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('class3.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()