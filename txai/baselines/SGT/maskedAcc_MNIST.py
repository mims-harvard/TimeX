
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
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',help='learning rate (default: 1.0)')

parser.add_argument('--maskAccType', default='trueMask', type=str)
parser.add_argument('--trainingType', default='regular', type=str)
parser.add_argument('--featuresDroped', type=float, default=0.1)
parser.add_argument('--RandomMasking', default=False, action="store_true", help='Random Masking while interpretable training')
parser.add_argument('--abs', default=False, action="store_true",help='take abs value of saliency while interpretable training')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='models', type=str)
parser.add_argument('--isMNIST', default=True, action="store_true",help='Dataset is MNIST')
parser.add_argument('--append', default='1', type=str)



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





    # Model
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


    filename=os.path.join(args.save_dir, saveFile+'model.th')
    checkpoint = torch.load(filename)




    drop=[0 ,10,20,30,40,50,60,70,80,90]


    columns=["Percentage"]  
    for d in drop:
        columns.append(str(d))

    saliencyMethods=["Grad","SG","IG", "DL", "GS","DLS","Random"]
    Grid=np.zeros((len(saliencyMethods),len(drop)+1),dtype=object)
    Grid[:,0]=saliencyMethods




    for s, saliencyMethod in enumerate(saliencyMethods):
            
        for dindx , d in enumerate(drop):
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            acc =regular.testForRoar(args,model,testloader,criterion,d/100.0,saliencyMethod)
            Grid[s,dindx+1]=acc

            print(saveFile,saliencyMethod,d , acc)
        print()



    resultFileName='./outputs/MaskedAcc/'+saveFile+"_"+args.maskAccType+".csv"
    Helper.save_intoCSV(Grid,resultFileName,col=columns)




if __name__ == '__main__':
    main()