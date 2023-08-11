import torch
import argparse
from functools import partial
import matplotlib.pyplot as plt

from txai.utils.data.preprocess import zip_x_time_y
from txai.utils.data import process_Synth, process_PAM, process_Epilepsy
from txai.synth_data.generate_spikes import SpikeTrainDataset
from txai.utils.data import EpiDataset#, PAMDataset

from txai.models.encoders.transformer_simple import TransformerMVTS
# from txai.models.gumbelmask_model import GumbelMask
# from txai.models.kumamask_model import TSKumaMask_TransformerPred as TSKumaMask


# Import all explainers:
from txai.baselines import TSR
from captum.attr import IntegratedGradients, Saliency
from txai.utils.baseline_comp.run_dynamask import run_dynamask

def get_explainer(key, args, device = None):

    key = key.lower()

    needs_training = False

    if key == 'fit':
        # Need to ensure we have generator
        needs_training = True
        pass

    elif key == 'dyna':
        explainer = partial(run_dynamask, device = device)

    elif key == 'tsr':
        def explainer(model, x, time, y):
            GradExplainer = Saliency(model)
            x = x.unsqueeze(0)
            time = time.transpose(0,1)
            out = TSR(GradExplainer, x, y, additional_forward_args = (time, None, True, False))
            return torch.from_numpy(out).to(device)

    elif key == 'ig':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            attr = IG.attribute(x, target = y, additional_forward_args = (time, None, True))
            return attr

    elif key == 'random':
        def explainer(model, x, time, y):
            return torch.randn_like(x).squeeze(1).float()

    elif key == 'attn':
        pass

    elif key == 'attngrad':
        pass

    elif key == 'attnnorm':
        pass

    elif key == 'model':
        def explainer(model, x, time, y):
            #model.eval()
            pred, mask = model(x.unsqueeze(1), time, captum_input = False)
            return mask

    #elif key == 'model_adv':
        #def model_adv(extractor, predictor, x, time, y):

    else:
        raise NotImplementedError('Cannot find explainer "{}"'.format(key))

    return explainer, needs_training

def get_dataset(data, split, device = None):
    '''
    Gets dataset based on only string entry for data and split given by number
    '''

    data = data.lower()

    if data == 'pam':
        train, val, test = process_PAM(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/PAMAP2data/', gethalf = True)
    
    elif (data == 'epi') or (data == 'epilepsy'):
        train, val, test = process_Epilepsy(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/Epilepsy/')

    elif (data == 'spike'):
        D = process_Synth(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/Spike/simple/')
        
        train = D['train_loader']
        val = D['val']
        test = D['test']

    return train, val, test