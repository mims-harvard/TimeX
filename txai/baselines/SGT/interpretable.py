

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torchvision
import random
import numpy as np
from .utils import progress_bar
import copy 
import torch.optim as optim
#from torchvision import datasets, transforms
import os
import sys 
import time
from . import Helper
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    InputXGradient,
    Saliency,
    NoiseTunnel
)
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
import numpy as np

use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def train(args,epoch,model,trainloader,optimizer,criterion,criterionKDL,Name=None):
    '''
    EXPECTS CAPTUM-STYLE INPUT OF B,T,D
    '''
    # if(Name==None):
    #     print('\nEpoch: %d' % epoch)
    # else:
    #     print('\nEpoch: %d' % epoch, Name)


    train_loss = 0
    Kl_loss=0
    Model_loss=0
    correct = 0
    correctAugmented=0
    total = 0
    model.train()
    softmax = nn.Softmax(dim=1)
    if(args.RandomMasking):
        maskType="randomMask"
    else:
        maskType="meanMask"
    

    #for batch_idx, (data, target) in enumerate(trainloader):
    for batch_idx, (data, time, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        if(args.featuresDroped!=0):
            model.eval()
            #numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

            tempData=data.clone()
            saliency = Saliency(model)
            # grads= saliency.attribute(data, target, abs=args.abs,
            #     additional_forward_args = (time, None, True)).mean(1).detach().cpu().to(dtype=torch.float)
            grads= saliency.attribute(data, target, abs=args.abs,
                additional_forward_args = (time, None, True)).detach().cpu().to(dtype=torch.float)

            for idx in range(tempData.shape[0]): # Goes across the batch
                singleMask=  Helper.get_SingleMask(args.featuresDroped,grads[idx], remove_important=False)
                # EDIT HERE:
                tempData[idx] = Helper.fill_SingleMask(tempData[idx],singleMask,maskType)


            maskedInputs=tempData.view(data.shape).detach()
            model.train()
        

       
        optimizer.zero_grad()
        output= model(data, time, captum_input = True)
        Modelloss = criterion(output, target)
        loss=Modelloss
        Model_loss+=Modelloss.item()


        if(args.featuresDroped!=0):
            maskedOutputs= model(maskedInputs, time, captum_input = True)
            if(args.isMNIST):
                KLloss = criterionKDL(maskedOutputs,output)

            else:
                maskedOutputs = F.log_softmax(maskedOutputs, dim=1)
                SoftmaxOutput=softmax(output)
                KLloss = criterionKDL(maskedOutputs,SoftmaxOutput)
           
            Kl_loss+=KLloss.item()
            loss=loss + KLloss

        else:
            KLloss=0


        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True) 
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()


        # if args.featuresDroped!=0 :
        
        #     progress_bar(batch_idx, len(trainloader), '# %.1f Loss: %.3f |  Modelloss %.3f  KLloss %.3f | Acc: %.3f'
        #          % (args.featuresDroped,train_loss/(batch_idx+1),Model_loss/(batch_idx+1), Kl_loss /(batch_idx+1), 100.*correct/total))
        # else:
        #     progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total ))


    return model , Model_loss , Kl_loss



def test(args,epoch,model,testloader,criterion,criterionKDL,best_acc,best_epoch,returnMaskedAcc=False):


    softmax = nn.Softmax(dim=1)

    model.eval()
    test_loss = 0
    Kl_loss=0
    correct = 0
    total = 0
    Model_loss=0
    correctAugmented=0
    augmentedAcc=0

    if(args.RandomMasking):
        maskType="randomMask"
    else:
        maskType="meanMask"

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True


            if(args.featuresDroped!=0):
                numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

                tempData=data.clone()
                saliency = Saliency(model)
                grads= saliency.attribute(data, target, abs=False).mean(1).detach().cpu().to(dtype=torch.float)
                if(args.abs):
                    grads=grads.abs()
                if(args.isMNIST):
                    tempData=tempData.view(tempData.shape[0], -1).detach()
                    tempGrads= grads.view(grads.shape[0], -1)
                    values,indx = torch.topk(tempGrads, numberOfFeatures, dim=1,largest=False)
                


                    for idx in range(tempData.shape[0]):
                        if args.RandomMasking:
                            min_=torch.min(tempData[idx]).item()
                            max_=torch.max(tempData[idx]).item()
                            randomMask = np.random.uniform(low=min_, high=max_, size=(len(indx[idx]),))
                            tempData[idx][indx[idx]]= torch.Tensor(randomMask).to(device)
                        else:
                            tempData[idx][indx[idx]]= data[0,0,0,0]
                else:
                             


                    for idx in range(tempData.shape[0]):
                        singleMask=  Helper.get_SingleMask(args.featuresDroped,grads[idx], remove_important=False)

                        tempData[idx] = Helper.fill_SingleMask(tempData[idx],singleMask,maskType)


                maskedInputs=tempData.view(data.shape).detach()
        

                maskedOutputs= model(maskedInputs)


            outputs= model(data)

            Modelloss = criterion(outputs, target)
            Model_loss+=Modelloss.item()
            test_loss += Modelloss.item()

            if(args.featuresDroped!=0):

                if(args.isMNIST):
                    KLloss = criterionKDL(maskedOutputs,outputs)
                else:
    
                    maskedOutputs = F.log_softmax(maskedOutputs, dim=1)
                    SoftmaxOutput=softmax(outputs)
                    KLloss = criterionKDL(maskedOutputs,SoftmaxOutput)



                Kl_loss+=KLloss.item()
                test_loss+=KLloss.item()



            predicted = outputs.argmax(dim=1, keepdim=True) 
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()




            if(args.featuresDroped!=0):
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f | best: Acc %.3f epoch %d'
                     % (test_loss/(batch_idx+1), 100.*correct/total,best_acc,best_epoch))
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f  | best: Acc %.3f epoch %d '
                         % (test_loss/(batch_idx+1), 100.*correct/total,best_acc,best_epoch))

    # Save checkpoint.
    acc = 100.*correct/total
    Kl_loss=Kl_loss/(batch_idx+1)
    
    if(returnMaskedAcc):
        return acc, augmentedAcc, Kl_loss
    else:
        return acc ,Kl_loss


