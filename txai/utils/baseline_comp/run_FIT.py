import torch

import sys, os
from tqdm import tqdm
from txai.baselines.FIT.TSX.explainers import FITExplainer
from txai.baselines.FIT.TSX.generator import JointFeatureGenerator

def run_FIT(
        model,
        X,
        time,
        FIT_obj, # FIT object with trained generator
        y = None,
    ):

    if y is None:
        model.eval()
        with torch.no_grad():
            y = model(X, time)

    score = FIT_obj.attribute(x = X, y = y, times = time)

    return score

def screen_FIT(
        model,
        test_tups,
        n_classes,
        generator = None,
        train_loader = None,
        val_loader = None,
        feature_size = 34, 
        generator_epochs = 50,
        skip_eval = False,
    ):
    '''
    
    '''

    FIT = FITExplainer(model = model, n_classes = n_classes)

    if generator is None:
        # Train generator
        generator_model = JointFeatureGenerator(
            feature_size = feature_size,
            prediction_size = 1,
            data = 'custom',
        )
        
        # Fit generator:
        FIT.fit_generator(
            generator_model,
            train_loader = train_loader,
            test_loader = val_loader,
            n_epochs = generator_epochs,
        )

    else:
        FIT.generator = generator

    all_exp = []

    if not skip_eval:
        for X, time, y in tqdm(test_tups):

            score = run_FIT(model, X, time, FIT, y = y)
            all_exp.append(score)

    return all_exp, FIT
    