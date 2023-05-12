

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import random
from utils import progress_bar
import copy 
import torch.optim as optim
from torchvision import datasets, transforms
import os
from . import Helper
import time


use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")





def train(args,epoch,model,trainloader,optimizer,criterion,Name=None):
    if(Name != None):
        print('\nEpoch: %d' % epoch,Name)
    else:
        print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    correctAugmented=0
    augmentedAcc=0
    total = 0
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True) 
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
                    
 
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




    return model


def test(args,epoch,model,testloader,criterion,best_acc,best_epoch,returnMaskedAcc=False,printing=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    correctAugmented=0
    augmentedAcc=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)


            outputs = model(inputs)
            loss = criterion(outputs, targets)
            

            test_loss += loss
            predicted = outputs.argmax(dim=1, keepdim=True) 
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()



            if(printing):
            
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) best Acc %.3f best epoch %d'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,best_acc,best_epoch))


    acc = 100.*correct/total
    
   
    if(returnMaskedAcc):
        return acc ,  augmentedAcc

    return acc




def testForRoar(args,model,testloader,criterion,percentage,saliencyMethod):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if(percentage>0):
                maskedData = Helper.getSalinecyAugmnetedBatch(model,inputs,targets,args.maskAccType,percentage,saliencyMethod)
            else:
                maskedData=inputs

            outputs = model(maskedData)
            loss = criterion(outputs, targets)
            

            test_loss += loss
            predicted = outputs.argmax(dim=1, keepdim=True) 
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()

    acc = 100.*correct/total
    
   


    return acc




