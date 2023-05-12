import numpy as np
import pandas as pd
from torch.autograd import Variable
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
import random
from  sklearn.preprocessing import minmax_scale
import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets

import time
import torch.nn.functional as F
import sys

import torch.utils.data as data_utils



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.set_printoptions(threshold=10_000)
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    InputXGradient,
    Saliency,
    NoiseTunnel
)

################################## General Helper Function ##############################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def load_CSV(file,returnDF=False,Flip=False):
    df = pd.read_csv(file)
    data=df.values
    if(Flip):
        print("Will Un-Flip before Loading")
        data=data.reshape((data.shape[1],data.shape[0]))
    if(returnDF):
        return df
    return data


def get_RandomMask(percentage,fattr_len):
    k = max(1, int(fattr_len*fattr_len * percentage))
    i = random.sample(range(0, fattr_len*fattr_len), k)
    mask = torch.zeros(fattr_len*fattr_len, dtype=torch.bool)
    mask[i]=True
    mask= mask.reshape(fattr_len,fattr_len)

    return mask



def get_SingleMask(percentage,fattr, remove_important=True,returnValues=False,Debug=False,numberOfFeatures=False):
    if(numberOfFeatures):
        k=percentage
    else:
        k = max(1, int(fattr.flatten().shape[0] * percentage))
        #k = max(1, int(len(fattr)*len(fattr) * percentage))
    if(Debug):
        print("numberOfFeatures", k, "remove_important",remove_important)
    #print(fattr.flatten().shape)
    #print('k', k)
    v, i = torch.topk(fattr.flatten(), k,largest=remove_important)
    mask = torch.zeros_like(fattr.flatten(), dtype=torch.bool)
    mask[i]=True
    mask= mask.reshape(fattr.shape)

    if(not returnValues):
        return mask
    else:
        return mask,v


def fill_SingleMask(input_image,mask,maskType="meanMask",customMask=None):
    channels = input_image.shape[0]
    num_masks = mask.sum() # Total number of values to fill
    #mask = mask.repeat(channels, 1, 1)

    if(maskType=="randomMask"):
        #mean_input=[]
        #for c in range(channels):
            # min_channel=torch.min(input_image[c,:,:]).item()
            # max_channel=torch.max(input_image[c,:,:]).item()
            # randomMask = np.random.uniform(low=min_channel, high=max_channel,size=(num_masks,))
            # mean_input.append(randomMask)
            # input_image[c,mask[c]] = torch.FloatTensor(randomMask).to(device)

        min_channel=torch.min(input_image).item()
        max_channel=torch.max(input_image).item()
        randomMask = np.random.uniform(low=min_channel, high=max_channel,size=(num_masks,))
        mean_input = (randomMask)
        input_image[mask] = torch.FloatTensor(randomMask).to(device) # Fill with mask values         
 
    else:
        if(maskType=="customMask"):
            mean_input=torch.Tensor(np.random.normal(0,1,input_image.mean([1, 2]).shape)).to(device)+customMask

        elif(maskType=="constantMask"):
            if(channels==3):
                mean_input=[customMask,customMask,customMask]
                mean_input=torch.FloatTensor(mean_input).to(device)
            else:
                mean_input=[customMask]
                mean_input=torch.FloatTensor(mean_input).to(device)
        elif(maskType=="trueMask"):
            mean_input = input_image[0,0,0]
        else:
            mean_input = input_image.mean([1, 2])
    
       
        # print(mean_input)
        input_image[mask] = mean_input.repeat(
            num_masks).reshape(num_masks, -1).T.flatten()
    return input_image






@torch.no_grad()
def getSalinecyAugmnetedBatch(model,images,target,maskType,percentage,method,mu=10):

    # images.requires_grad=True
    images = Variable(images,  volatile=False, requires_grad=True)


    if(method=="Grad"):
        attr = Saliency(model)
        saliency= attr.attribute(images, target).mean(1).detach().cpu().to(
            dtype=torch.float)
        saliency=saliency.abs()

    elif(method=="IG"):
        attr = IntegratedGradients(model)
        baseline_single=torch.Tensor(np.random.random(images.shape)).to(device)
        saliency= attr.attribute(images,baselines=baseline_single,target=target).mean(1).detach().cpu().to(
            dtype=torch.float)
        saliency=saliency.abs()

    elif(method=="DL"):
        attr = DeepLift(model)
        baseline_single=torch.Tensor(np.random.random(images.shape)).to(device)
        saliency= attr.attribute(images,baselines=baseline_single,target=target).mean(1).detach().cpu().to(
            dtype=torch.float)
        saliency=saliency.abs()

    elif(method=="GS"):

        attr = GradientShap(model)
        baseline_multiple=torch.Tensor(np.random.random((images.shape[0]*5,images.shape[1],images.shape[2],images.shape[3]))).to(device)
        baseline_multiple.requires_grad=True


        saliency = attr.attribute(images,baselines=baseline_multiple,stdevs=0.09,target=target).mean(1).detach().cpu().to(
            dtype=torch.float)
        saliency=saliency.abs()

    elif(method=="DLS"):
        attr = DeepLiftShap(model)
       
        baseline_multiple=torch.Tensor(np.random.random((images.shape[0]*5,images.shape[1],images.shape[2],images.shape[3]))).to(device)
        baseline_multiple.requires_grad=True
        saliency = attr.attribute(images,baselines=baseline_multiple,target=target).mean(1).detach().cpu().to(
            dtype=torch.float)
        saliency=saliency.abs()
    elif(method=="SG"):
        Grad_ = Saliency(model)
        attr = NoiseTunnel(Grad_)
        saliency= attr.attribute(images, nt_type='smoothgrad_sq', target=target).mean(1).detach().cpu().to(
                dtype=torch.float)
        saliency=saliency.abs()


    
    tempData=images.clone()

    for _ in range(tempData.shape[0]):
        if(method=="Random"):
            mask =get_RandomMask(percentage,images.shape[2])
        else:
            mask = get_SingleMask(percentage,saliency[_], remove_important=True)
        tempData[_]= fill_SingleMask(tempData[_],mask,maskType=maskType,customMask=mu)
    return tempData




def save_intoCSV(data,file,Flip=False,col=None,index=False):
    if(Flip):
        print("Will Flip before Saving")
        data=data.reshape((data.shape[1],data.shape[0]))


    df = pd.DataFrame(data)
    if(col!=None):
        df.columns = col
    df.to_csv(file,index=index)




def  getSaliency(model,testloader,method,NumberOfSamples,abs=True):
    model.eval()
    if(method=="Grad"):
        Grad = Saliency(model)

    elif(method=="IG"):
        IG = IntegratedGradients(model)

    elif(method=="DL"):
        DL = DeepLift(model)

    elif(method=="GS"):
        GS = GradientShap(model)
    elif(method=="DLS"):
        DLS = DeepLiftShap(model)
    elif(method=="SG"):

        Grad_ = Saliency(model)
        SG = NoiseTunnel(Grad_)


    dataiter = iter(testloader)
    images, labels = dataiter.next()


    indx=0
    Output=torch.zeros((NumberOfSamples,images.shape[1],images.shape[2],images.shape[3]))
    allData=torch.zeros((NumberOfSamples,images.shape[1],images.shape[2],images.shape[3]))
    labels= torch.zeros((NumberOfSamples,))
    for i, (data, target) in enumerate(testloader):

        if(i%10==0):
            print(i,"of",len(testloader),"getSaliency",method)

        data, target = data.to(device), target.to(device)
        baseline=torch.Tensor(np.random.random((1,data.shape[1],data.shape[2],data.shape[3]))).to(device)
        input = Variable(data,  volatile=False, requires_grad=True)
        if(method=="Grad"):
            attributions = Grad.attribute(input,target=target,abs=False)
        if(method=="IG"):
            attributions = IG.attribute(input, baselines=baseline,  target=target)
        if(method=="DL"):
            attributions = DL.attribute(input,  baselines=baseline,target=target)



        baseline=torch.Tensor(np.random.random((input.shape[0]*5,input.shape[1],input.shape[2],data.shape[3]))).to(device)

        if(method=="GS"):
            attributions = GS.attribute(input, baselines=baseline, stdevs=0.09,target=target)


        if(method=="DLS"):
            attributions = DLS.attribute(input, baselines=baseline, target=target)




        if(method=="SG"):
            attributions = SG.attribute(input,target=target)


        if(abs):
            saliency = torch.abs(attributions).detach()
        else:
            saliency=attributions.detach()



        # if(i==0):
        #     print(saliency[2])
        Output[indx:indx+saliency.shape[0],:,:,:]=saliency.detach()



        labels[indx:indx+saliency.shape[0]]=target.detach()
        allData[indx:indx+saliency.shape[0]]=data.detach()




        indx+=saliency.shape[0]

    torch.cuda.empty_cache() 
    return Output , labels,allData
