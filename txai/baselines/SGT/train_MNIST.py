
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import random
import os
import argparse
import numpy as np
import importlib
from .cnn import Net
import torch.optim as optim
from torchvision import datasets, transforms
import time

import .Helper
import .regular
import .interpretable 
from .Helper import save_checkpoint



parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 14)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',help='learning rate (default: 1.0)')


parser.add_argument('--trainingType', default='regular', type=str)
parser.add_argument('--featuresDroped', type=float, default=0.1)
parser.add_argument('--RandomMasking', default=False, action="store_true", help='Random Masking while interpretable training')
parser.add_argument('--abs', default=False, action="store_true",help='take abs value of saliency while interpretable training')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='models', type=str)
parser.add_argument('--isMNIST', default=True, action="store_true",help='Dataset is MNIST')
parser.add_argument('--append', default='1', type=str)
parser.add_argument('--patience', default='20', type=str)

# torch.manual_seed(args.seed)


def main():
    args = parser.parse_args()
    start_epoch = 0 
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    print('==> Preparing data..')

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    dataset1 = datasets.MNIST('./data/MNIST_data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/MNIST_data', train=False,
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    testloader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    print('==> Building model..')
    model = Net().to(device)




    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    criterionKDL=torch.nn.KLDivLoss(log_target=True,reduction = 'batchmean')

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    saveFile= "MNIST_"+args.trainingType


            
    if("interpretable" in args.trainingType):
        saveFile=saveFile+"featuresDroped_"+str(args.featuresDroped)+"_"
        if(args.RandomMasking):
            saveFile=saveFile+"RandomMasking_"        
        if(args.abs):
            saveFile=saveFile+"abs_"

    saveFile=saveFile+args.append+"_"


    best_prec1=0
    best_epoch=0
    NoChange=0
    start_training=time.time()
    if(args.resume):

        filename=os.path.join(args.save_dir, saveFile+'model.th')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch=checkpoint['epoch']
        best_prec1=checkpoint['best_prec1']
    for epoch in range(start_epoch,args.epochs ):
        start_epoch_time=time.time()

        if(args.trainingType=="regular"):
            model= regular.train(args,epoch,model,trainloader,optimizer,criterion,saveFile)

            prec1 =regular.test(args,epoch,model,testloader,criterion,best_prec1,best_epoch)

        elif(args.trainingType=="interpretable"):
            model , train_Model_loss , train_Kl_loss = interpretable.train(args,epoch,model,trainloader,optimizer,criterion,criterionKDL,Name=saveFile)

            prec1 ,test_Kl_loss  = interpretable.test(args,epoch,model,testloader,criterion,criterionKDL,best_prec1,best_epoch)

        end_epoch_time=time.time()
        print("epoch time:",end_epoch_time-start_epoch_time,"No Change Flag",NoChange)


        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if(is_best):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'model.th'))
            NoChange=0
        else:
            NoChange+=1

        if(epoch+1 ==args.epochs):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'Last_model.th'))
        end_epoch_time=time.time()


        if(NoChange>=args.patience):
            best_epoch=epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'epoch': epoch,
            }, is_best, filename=os.path.join(args.save_dir, saveFile+'Last_model.th'))
            break

    end_training=time.time()
    print("Trainig time", end_training- start_training)

if __name__ == '__main__':
    main()