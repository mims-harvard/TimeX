import pathlib

import numpy as np
import torch
from sklearn import preprocessing

from .Helper import givenAttGetRescaledSaliency
from .Plotting.plot import plotExampleBox


def get_attribution(saliency, input, label, baselines, feature_mask, sliding_window_shape):
    if baselines is None:
        return saliency.attribute(input, target=label).data.cpu().numpy()
    else:
        if feature_mask is not None:
            return saliency.attribute(input, baselines=baselines, target=label, feature_mask=feature_mask).data.cpu().numpy()
        elif sliding_window_shape is not None:
            return saliency.attribute(input, sliding_window_shapes=sliding_window_shape, baselines=baselines, target=label).data.cpu().numpy()
        else:
            return saliency.attribute(input, baselines=baselines, target=label).data.cpu().numpy()


def getTwoStepRescaling(saliency, input, TestingLabel, hasBaseline=None, hasFeatureMask=None,
                        hasSliding_window_shapes=None, return_time_ft_contributions=False, ft_dim_last=True):
    batch_size, sequence_length, input_size = input.shape if ft_dim_last else (input.shape[0], input.shape[2], input.shape[1])
    assignment = input[0, 0, 0]
    timeGrad = np.zeros((batch_size, sequence_length))
    inputGrad = np.zeros((batch_size, input_size))
    newGrad = np.zeros(input.shape)

    ActualGrad = get_attribution(saliency, input, TestingLabel, hasBaseline, hasFeatureMask, hasSliding_window_shapes)

    timeGrad[:] = np.mean(np.absolute(ActualGrad), axis=2 if ft_dim_last else 1)

    # for t in range(sequence_length):
    #     newInput = input.clone()
    #     if ft_dim_last:
    #         newInput[:,t,:]=assignment
    #     else:
    #         newInput[:,:,t]=assignment
    #
    #     timeGrad_perTime = get_attribution(saliency, newInput, TestingLabel, hasBaseline, hasFeatureMask, hasSliding_window_shapes)
    #     timeGrad_perTime= np.absolute(ActualGrad - timeGrad_perTime)
    #     timeGrad[:,t] = np.sum(timeGrad_perTime, axis=(1, 2))

    timeContribution = preprocessing.minmax_scale(timeGrad, axis=1)
    # meanTime = np.quantile(timeContribution, .55)

    time_contributions = np.zeros((batch_size, sequence_length, input_size))
    time_contributions[:, :] = timeContribution[:, :, None]
    feature_contributions = np.zeros((batch_size, sequence_length, input_size))

    for t in range(sequence_length):
        # TODO: Improve performance by only computing ft contribution if above alpha threshold
        for c in range(input_size):
            newInput = input.clone()
            i1, i2 = (t, c) if ft_dim_last else (c, t)
            newInput[:, i1, i2] = assignment

            inputGrad_perInput = get_attribution(saliency, newInput, TestingLabel, hasBaseline, hasFeatureMask, hasSliding_window_shapes)
            inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
            inputGrad[:, c] = np.sum(inputGrad_perInput, axis=(1, 2))
        featureContribution = preprocessing.minmax_scale(inputGrad, axis=-1)
        feature_contributions[:, t, :] = featureContribution

        alpha = 0

        for c in range(input_size):
            for batch in range(batch_size):
                i1, i2 = (t, c) if ft_dim_last else (c, t)
                newGrad[batch, i1, i2] = timeContribution[batch, t] * featureContribution[batch, c] if timeContribution[batch, t] > alpha else 0

    return newGrad, time_contributions, feature_contributions if return_time_ft_contributions else newGrad


def get_tsr_saliency(saliency, input, labels, baseline=None, inputs_to_graph=(), graph_dir=None,
                     graph_name='TSR', cur_batch=None, ft_dim_last=True):
    batch_size, num_timesteps, num_features = input.shape

    TSR_attributions, time_contributions, ft_contributions = getTwoStepRescaling(saliency, input, labels, hasBaseline=baseline,
                                                                                 return_time_ft_contributions=True, ft_dim_last=ft_dim_last)

    assert len(inputs_to_graph) == 0 or (graph_dir is not None and cur_batch is not None)
    for index in inputs_to_graph:
        index_within_batch = index - batch_size * cur_batch
        if 0 <= index_within_batch < batch_size:
            pathlib.Path(graph_dir).mkdir(parents=True, exist_ok=True)
            plotExampleBox(TSR_attributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_attr', greyScale=True, flip=True)
            plotExampleBox(time_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_time_cont', greyScale=True, flip=True)
            plotExampleBox(ft_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_ft_cont', greyScale=True, flip=True)

    return givenAttGetRescaledSaliency(num_timesteps, num_features, TSR_attributions, isTensor=False)
