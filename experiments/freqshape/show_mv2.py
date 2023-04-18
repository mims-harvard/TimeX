import torch

from txai.vis.visualize_mv2 import visualize
from txai.models.modelv2 import Modelv2
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv2
from txai.synth_data.simple_spike import SpikeTrainDataset

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    D = process_Synth(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    spath = 'models/Scomb_v2_split=1.pt'
    print('Loading model at {}'.format(spath))

    sdict, config = torch.load(spath)
    print('Config:\n', config)

    model = Modelv2(**config)
    model.to(device)

    model.load_state_dict(sdict)

    f1, _ = eval_mv2(test, model)
    print('Test F1: {:.4f}'.format(f1))

    visualize(model, test, show = False, class_num = 2)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig('mah_class3.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()