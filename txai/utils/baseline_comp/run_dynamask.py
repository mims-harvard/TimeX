import torch
from tqdm import tqdm
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../baselines'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../baselines/Dynamask'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../baselines/Dynamask/attribution'))

from txai.baselines.Dynamask.attribution.mask import Mask
from txai.baselines.Dynamask.attribution.perturbation import GaussianBlur
from txai.baselines.Dynamask.utils.losses import cross_entropy

default_config = {
    'keep_ratio': 0.1,
    'initial_mask_coeff': 0.5,
    'size_reg_factor_init': 0.01,
    'size_reg_factor_dilation': 100,
    'time_reg_factor': 0,
    'learning_rate': 1.0e-1,
    'momentum': 0.9,
}

def mse(Y, Y_target):
    """
    Computes the mean squared error between Y and Y_target.
    From Dynamask codebase (https://github.com/JonathanCrabbe/Dynamask/blob/main/utils/losses.py)
    """
    return torch.mean((Y - Y_target) ** 2)

def run_dynamask(
        model,
        X,
        time_input,
        n_epoch=50,
        keep_ratio = 0.01, #  Fraction of elements in X that should be kept by the mask (called a in the paper).
        initial_mask_coeff = 0.5, # Always 0.5 in paper
        size_reg_factor_init = 0.001,  # Initial coefficient for the regulator part of the total loss (lambda_0)
        size_reg_factor_dilation = 100_000, # Ratio between the final and the initial size regulation factor (called delta in the paper).
        time_reg_factor = 0, # Regulation factor for the variation in time (called lambda_c in the paper).
        learning_rate = 1.0e-1,
        momentum = 0.9,
        y = None,
        device = None,):
    '''
    Per Dynamask paper:
        X: Input matrix Txd torch tensor (i.e. no batch)
        f: Black-box model
        y (target): Can specify label manually
    '''

    pert = GaussianBlur(device)
    mask = Mask(pert, device, task="classification")

    CE = torch.nn.CrossEntropyLoss()

    mask.fit(X, time_input, model, loss_function = CE, 
        target = y,
        n_epoch=n_epoch,
        keep_ratio = keep_ratio,
        initial_mask_coeff = initial_mask_coeff,
        size_reg_factor_init = size_reg_factor_init,
        size_reg_factor_dilation = size_reg_factor_dilation,
        time_reg_factor = time_reg_factor,
        learning_rate = learning_rate,
        momentum = momentum 
        )

    # Extract mask tensor from model:
    return mask.mask_tensor


def screen_dynamask(
        model,
        test_tuples, 
        only_correct = True,
        device = None,
        dynamask_config = default_config):
    '''
    Screens over an entire test set to produce explanations for Dynamask Explainer

    - Assumes all input tensors are on same device

    test_tuples: list of tuples
        - [(X_0, time_0, y_0), ..., (X_N, time_N, y_N)]
    '''

    out_masks = []

    model.eval()
    for X, time, y in test_tuples:

        time.requires_grad_ = False
    
        #with torch.no_grad():
        #     out = model(X, time)

        #if (out.softmax(dim=1).argmax(dim=1) == y) or (not only_correct):

            # Assumes no transformation of shapes, etc.
        mask = run_dynamask(model = model, X = X, time_input = time, **default_config, device = device)

        out_masks.append(mask)

    return out_masks



    