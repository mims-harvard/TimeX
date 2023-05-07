import torch
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations

# Models:
from txai.models.modelv6 import Modelv6
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.modelv6_v2_ptnew import Modelv6_v2_PT
from txai.models.modelv6_v3 import Modelv6_v3

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG

from txai.utils.functional import cosine_sim_matrix

'''
Simply look at the linear layer
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    # Loading:
    print('Loading model at {}'.format(args.model_path))
    sdict, config = torch.load(args.model_path)
    print('Config:\n', config)

    model = Modelv6_v2_PT(**config)
    model.load_state_dict(sdict)
    model.eval()
    model.to(device)

    w = model.ptype_predictor_net.weight

    print('weights', w.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()

    main(args)