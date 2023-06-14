import argparse
import torch
import torch.nn as nn
import sys
from tqdm import trange
from captum.attr import Saliency
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.trainers.train_transformer import train
from txai.baselines.SGT.SGT import SGT_train
from txai.baselines.SGT import Helper as Helper
from txai.utils.predictors.loss import Poly1CrossEntropyLoss # Get Polyloss
from txai.utils.data import process_Synth
from txai.utils.data.datasets import DatasetwInds
from txai.synth_data.simple_spike import SpikeTrainDataset
import numpy as np
from txai.utils.evaluation import ground_truth_xai_eval

device = 'cuda'
import torch.nn.functional as F
class Args():
    RandomMasking = True # Fills mask with random values corresponding to each sample
    abs = True
    featuresDroped = 0.9 

# parser = argparse.ArgumentParser()
# parser.add_argument('--split', type = int)
# ARGS = parser.parse_args()
i = 4

#for i in range(1, 6):
print(f'\n------------------ Split {i} ------------------')
# D = process_Synth(split_no = i, device = device, base_path = '/home/huan/Documents/TimeSeriesCBM-main/FreqShape')
# train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

# val, test = D['val'], D['test']

# model = TransformerMVTS(
#     d_inp = val[0].shape[2],
#     max_len = val[0].shape[0],
#     n_classes = 2,
#     no_return_attn = True
# )

D = process_Synth(split_no = i, device = device, base_path = '/home/huan/Documents/TimeSeriesCBM-main/SeqCombMV')
dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

val, test = D['val'], D['test']

model = TransformerMVTS(
    d_inp = val[0].shape[-1],
    max_len = val[0].shape[0],
    n_classes = 4,
    trans_dim_feedforward = 128,
    nlayers = 2,
    trans_dropout = 0.25,
    d_pe = 16,
)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0.01)

args = Args()
nepochs = 10

for epoch in range(nepochs):
    print(epoch)
    # model, loss, auc = SGT_train(
    #     args = args,
    #     epoch = epoch,
    #     model = model,
    #     trainloader = train_loader,
    #     optimizer = optimizer,
    criterion =  Poly1CrossEntropyLoss(
        num_classes = 4,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
        )
    criterionKDL = torch.nn.KLDivLoss(log_target = False, reduction = 'batchmean')
    #     Name = None,
    # )
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
    for batch_idx, (data, time, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        model.eval()
        #numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

        tempData=data.clone()
        saliency = Saliency(model)
        # grads= saliency.attribute(data, target, abs=args.abs,
        #     additional_forward_args = (time, None, True)).mean(1).detach().cpu().to(dtype=torch.float)
        grads= saliency.attribute(data, target, abs=args.abs,
            additional_forward_args = time).detach().cpu().to(dtype=torch.float)

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
        maskedOutputs= model(maskedInputs, time, captum_input = True)
        maskedOutputs = F.log_softmax(maskedOutputs, dim=1)
        SoftmaxOutput=softmax(output)
        KLloss = criterionKDL(maskedOutputs,SoftmaxOutput)
        Kl_loss+=KLloss.item()
        loss=loss + KLloss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True) 
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

target = test[2]
data = test[0]
time = test[1]
gt_exps = D['gt_exps']
# z_seq = model.embed(X, times, captum_input = False, aggregate = False)
model.eval()
#numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

tempData = data.clone()
saliency = Saliency(model)
# grads= saliency.attribute(data, target, abs=args.abs,
#     additional_forward_args = (time, None, True)).mean(1).detach().cpu().to(dtype=torch.float)
grads= saliency.attribute(data, target, abs=args.abs,
    additional_forward_args =time).detach().cpu().to(dtype=torch.float)

for idx in range(tempData.shape[0]): # Goes across the batch
    singleMask=  Helper.get_SingleMask(args.featuresDroped,grads[idx],\
                                        remove_important=False)
    tempData[idx] = Helper.fill_SingleMask(tempData[idx],singleMask,maskType)


maskedInputs=tempData.view(data.shape).detach()
# mask_in, ste_mask = decoder(z_seq, X, times)
results_dict = ground_truth_xai_eval(maskedInputs[:,(target != 0).detach().cpu(),:], \
                                     gt_exps[:,(target != 0).detach().cpu(),:])


for k, v in results_dict.items():
    print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))