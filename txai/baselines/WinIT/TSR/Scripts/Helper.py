import itertools
import time

import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
from  sklearn.preprocessing import minmax_scale
import torch.nn.functional as F
import torch.utils.data as data_utils
import timesynth as ts

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################## General Helper Function ##############################


def load_CSV(file,returnDF=False,Flip=False):
	df = pd.read_csv(file)
	data=df.values
	if(Flip):
		print("Will Un-Flip before Loading")
		data=data.reshape((data.shape[1],data.shape[0]))
	if(returnDF):
		return df
	return data


def givenAttGetRescaledSaliency(num_timesteps, num_features, attributions, isTensor=True):
    if(isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    saliency=saliency.reshape(-1, num_timesteps * num_features)
    rescaledSaliency=minmax_scale(saliency,axis=1)
    rescaledSaliency=rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency


def getSaliencyMethodsFromArgs(args):
    saliency_methods = []
    if args.GradFlag:
        saliency_methods.append("Grad")
    if args.GradTSRFlag:
        saliency_methods.append("Grad_TSR")
    if args.IGFlag:
        saliency_methods.append("IG")
    if args.IGTSRFlag:
        saliency_methods.append("IG_TSR")
    if args.DLFlag:
        saliency_methods.append("DL")
    if args.GSFlag:
        saliency_methods.append("GS")
    if args.DLSFlag:
        saliency_methods.append("DLS")
    if args.DLSTSRFlag:
        saliency_methods.append("DLS_TSR")
    if args.SGFlag:
        saliency_methods.append("SG")
    if args.ShapleySamplingFlag:
        saliency_methods.append("ShapleySampling")
    if args.FeaturePermutationFlag:
        saliency_methods.append("FeaturePermutation")
    if args.FeatureAblationFlag:
        saliency_methods.append("FeatureAblation")
    if args.OcclusionFlag:
        saliency_methods.append("Occlusion")
    if args.FITFlag:
        saliency_methods.append("FIT")
    if args.IFITFlag:
        saliency_methods.append("IFIT")
    if args.WFITFlag:
        saliency_methods.append("WFIT")
    if args.IWFITFlag:
        saliency_methods.append("IWFIT")
    return saliency_methods


def save_intoCSV(data,file,Flip=False,col=None,index=False):
	if(Flip):
		print("Will Flip before Saving")
		data=data.reshape((data.shape[1],data.shape[0]))


	df = pd.DataFrame(data)
	if(col!=None):
		df.columns = col
	df.to_csv(file,index=index)



def getIndexOfXhighestFeatures(array,X):
    return np.argpartition(array, int(-1*X))[int(-1*X):]


def getAverageOfMaxX(array,X):
    index = getIndexOfXhighestFeatures(array,X)
    avg=np.mean(array[index])
    return avg


def getIndexOfAllhighestSalientValues(array,percentageArray):
    X=array.shape[0]
    # index=np.argpartition(array, int(-1*X))
    index=np.argsort(array)
    totalSaliency=np.sum(array)
    indexes=[]
    X=1
    for percentage in percentageArray:
        actualPercentage=percentage/100

        index_X=index[int(-1*X):]

        percentageDroped=np.sum(array[index_X])/totalSaliency
        if(percentageDroped<actualPercentage):
            X_=X+1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X])/totalSaliency
            if(not (percentageDroped_>actualPercentage)):
                while(percentageDroped<actualPercentage and X<array.shape[0]-1):
                    X=X+1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency
        elif(percentageDroped>actualPercentage):
            X_=X-1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X_])/totalSaliency
            if(not (percentageDroped_<actualPercentage)):

                while(percentageDroped>actualPercentage and X>1):
                    X=X-1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency

        indexes.append(index_X)
    return indexes


def generateNewSample(data_generation_process, num_timesteps, num_features, sampler, frequency, kernel, ar_param, order, has_noise):
    if data_generation_process is None:
        sample=np.random.normal(0,1,[num_timesteps, num_features])

    else:
        time_sampler = ts.TimeSampler(stop_time=20)
        sample=np.zeros([num_timesteps, num_features])


        if sampler == "regular":
            time = time_sampler.sample_regular_time(num_points=num_timesteps*2, keep_percentage=50)
        else:
            time = time_sampler.sample_irregular_time(num_points=num_timesteps*2, keep_percentage=50)


        for i in range(num_features):
            if data_generation_process == "Harmonic":
                 signal = ts.signals.Sinusoidal(frequency=frequency)

            elif data_generation_process == "GaussianProcess":
                signal = ts.signals.GaussianProcess(kernel=kernel, nu=3./2)

            elif data_generation_process == "PseudoPeriodic":
                signal = ts.signals.PseudoPeriodic(frequency=frequency, freqSD=0.01, ampSD=0.5)

            elif data_generation_process == "AutoRegressive":
                signal = ts.signals.AutoRegressive(ar_param=[ar_param])

            elif data_generation_process == "CAR":
                signal = ts.signals.CAR(ar_param=ar_param, sigma=0.01)

            elif data_generation_process == "NARMA":
                signal = ts.signals.NARMA(order=order)

            if has_noise:
                noise= ts.noise.GaussianNoise(std=0.3)
                timeseries = ts.TimeSeries(signal, noise_generator=noise)
            else:
                timeseries = ts.TimeSeries(signal)

            feature, signals, errors = timeseries.sample(time)
            sample[:,i]= feature
    return sample


def maskData(data_generation_process, num_timesteps, num_features, sampler, frequency, kernel, ar_param, order,
             has_noise, data, mask, noise=False):
    newData= np.zeros((data.shape))
    if(noise):
        noiseSample= generateNewSample(data_generation_process, num_timesteps, num_features, sampler, frequency, kernel, ar_param, order, has_noise)
        noiseSample=noiseSample.reshape(data.shape[1])
    for i in range(mask.shape[0]):
        newData[i,:]=data[i,:]
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        if(noise):
            newData[i,cleanIndex]=noiseSample[cleanIndex]
        else:
            newData[i,cleanIndex]=0

    return newData



def getRowColMaskIndex(mask,rows,columns):
    InColumn=np.zeros((mask.shape[0],columns),dtype=object)
    InRow=np.zeros((mask.shape[0],rows),dtype=object)
    InColumn[:,:]=False
    InRow[:,:]=False
    for i in range(mask.shape[0]):
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        for index in range(cleanIndex.shape[0]):
            InColumn[i,cleanIndex[index]%columns]=True
            InRow[i,int(cleanIndex[index]/columns)]=True
    return InRow,InColumn






################################## Get Accuracy Functions ##############################



def checkAccuracy(test_loader, model, num_timesteps, num_features, isCNN=False, returnLoss=False):

    model.eval()

    correct = 0
    total = 0
    if(returnLoss):
        loss=0
        criterion = nn.CrossEntropyLoss()

    for (samples, labels) in test_loader:
        if isCNN:
            samples = torch.unsqueeze(samples, dim=1).to(device)

        outputs = model(samples)
        if(returnLoss):
            labels = labels.to(device)
            loss+=criterion(outputs, labels).data


        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    if(returnLoss):
        loss=loss/len(test_loader)
        return  (100 * float(correct) / total),loss

    return  (100 * float(correct) / total)
