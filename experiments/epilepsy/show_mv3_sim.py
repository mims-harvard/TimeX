import torch

from txai.vis.visualize_mv3 import visualize
from txai.models.modelv3 import Modelv3EmbSim
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv3_sim
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_Epilepsy

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    trainEpi, val, test = process_Epilepsy(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')

    test_tuple = (test.X, test.time, test.y)

    spath = 'models/v3sep_split=1.pt'
    print('Loading model at {}'.format(spath))

    sdict, config = torch.load(spath)
    print('Config:\n', config)

    model = Modelv3EmbSim(**config)
    model.to(device)

    model.load_state_dict(sdict)

    f1, _ = eval_mv3_sim(test_tuple, model)
    print('Test F1: {:.4f}'.format(f1))

    visualize(model, test_tuple, show = False, class_num = 1, sim = True)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig('mv3_sim_class1.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()