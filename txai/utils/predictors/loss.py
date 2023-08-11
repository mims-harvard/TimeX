import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def exp_criterion_evaluation(mask: torch.Tensor, beta: float, exp_criterion: torch.nn.Module):

    if not (isinstance(beta, torch.Tensor) and (isinstance(exp_criterion, list))):
        l = exp_criterion(mask)
        return beta * l, [l.item()]

    # Else, need list-based evaluation
    llist = []
    for i in range(len(beta)):
        
        l = exp_criterion[i](mask)
        llist.append(l.item())

        if i == 0:
            lsum = beta[i] * l
        else:
            lsum += beta[i] * l

    return lsum, llist

# def exp_criterion_evaluation_permask(mask_list, beta, exp_criterion):

#     if not (isinstance(beta, torch.Tensor) and (isinstance(exp_criterion, list))):
#         l = exp_criterion(mask)
#         return beta * l, [l.item()]
    
#     # Else, need list-based evaluation
#     llist = []
#     for i in range(len(beta)):
        
#         l = exp_criterion[i](mask)
#         llist.append(l.item())

#         if i == 0:
#             lsum = beta[i] * l
#         else:
#             lsum += beta[i] * l

#     return lsum, llist

def gini_loss(x):
    '''
    Assumes input is of size (N,), i.e. one-dimensional
    '''
    
    gini = (x.view(-1,1) - x.repeat(x.shape[0],1)).abs().sum()

    #gini /= (2 * (x.shape[0] ** 2) * x.mean() + 1e-9)
    gini /= (2 * (x.shape[0] ** 2) + 1e-9)

    return gini

# cite: https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        #print('logits', logits)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1

class SATLoss(nn.Module):

    def __init__(self, criterion, 
            fix_r = True, 
            init_r = 0.9, 
            beta = 0.5,
            final_r = 0.1,
            decay_interval = None,
            decay_r = None):
        '''
        Provides a wrapper around a classification criterion that uses GSAT loss principle

        criterion: base classification loss function
        fix_r: bool, if True, fixes the r value during training
        init_r: initial r value, if fix_r==True, init_r used as the fixed r value
        '''
        super(SATLoss, self).__init__()

        self.criterion = criterion
        self.init_r  = init_r
        self.beta = beta
        self.final_r = final_r
        self.fix_r = fix_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = self.init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def forward(self, att, clf_logits, clf_labels, epoch = None):
        '''
        Params:
            att: p_uv as referred to in paper; outputs of SAT attention mechanisms
            clf_logits: output of classification head of model
            clf_labels: ground-truth labels for classification 
            epoch: Don't set if using fixed r value
        '''
        # print('clf logits', clf_logits)
        # print('clf_labels', clf_labels)
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.init_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        # att shape shouldn't matter - don't need to flatten
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        #pred_loss = pred_loss # Took away pred_loss_coef because it's redundant with beta
        #info_loss = info_loss * self.info_loss_coef
        info_loss = info_loss * self.beta
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

class SATGiniLoss(nn.Module):

    def __init__(self, criterion, beta = 0.5):
        super(SATGiniLoss, self).__init__()

        self.criterion = criterion
        self.beta = beta

    def forward(self, att, clf_logits, clf_labels):

        pred_loss = self.criterion(clf_logits, clf_labels)
        # pred_loss - beta * gini_loss

        # Must handle over batches:
        info_loss = torch.sum(torch.stack([gini_loss(torch.log(att[i] + 1e-6)) for i in range(att.shape[0])]))

        loss = pred_loss - self.beta * info_loss
        loss_dict = {'loss':loss.item(), 'pred':pred_loss.item(), 'info': info_loss.item()}
        #print(loss_dict)
        return loss, loss_dict

class GiniLoss(nn.Module):

    def __init__(self):
        super(GiniLoss, self).__init__()

    def forward(self, att):
        # Must handle over batches:
        loss = -1.0 * torch.sum(torch.stack([gini_loss(att[i] + 1e-6) for i in range(att.shape[0])]))
        return loss

class L1Loss(nn.Module):
    def __init__(self, diff = None, norm = False):
        super(L1Loss, self).__init__()
        self.diff = 0 if diff is None else diff
        self.norm = norm

    def forward(self, logits):
        #loss = torch.sum(torch.abs(torch.sum(torch.abs(attn), dim=1) - self.diff))
        if self.norm:
            l = logits.sum() / (logits.flatten().shape[0])
            #print(l)
            return l

        return logits.sum() * (1 / logits.shape[0]) 

class L1Loss_permask(nn.Module):
    def __init__(self, norm = False):
        super(L1Loss_permask, self).__init__()
        self.norm = norm

    def forward(self, logits):
        for i in range(len(logits)):
            if self.norm:
                if i == 0:
                    l = logits[i].sum() / (logits[i].flatten().shape[0])
                else:
                    l += logits[i].sum() / (logits[i].flatten().shape[0])
        return l

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, attn):
        loss = torch.sum(torch.pow(attn, 2)).sqrt()
        return loss

class GSATLoss(nn.Module):

    def __init__(self, r):
        super(GSATLoss, self).__init__()
        self.r = r

    def forward(self, att):
        if torch.any(torch.isnan(att)):
            print('ALERT - att has nans')
            exit()
        if torch.any(att < 0):
            print('ALERT - att less than 0')
            exit()
        assert (att < 0).sum() == 0
        info_loss = (att * torch.log(att/self.r + 1e-6) + (1-att) * torch.log((1-att)/(1-self.r + 1e-6) + 1e-6)).mean()
        ##print(info_loss)
        if torch.any(torch.isnan(info_loss)):
            print('INFO LOSS NAN')
            exit()
        return info_loss

class GSATLoss_Extended(nn.Module):

    def __init__(self, r):
        super(GSATLoss_Extended, self).__init__()
        self.r = r

    def forward(self, src, times, smoother_stats, att):
        if torch.any(torch.isnan(att)):
            print('ALERT - att has nans')
            exit()
        if torch.any(att < 0):
            print('ALERT - att less than 0')
            exit()
        info_loss = (att * torch.log(att/(self.r + 1e-6) + 1e-6) + (1-att) * torch.log((1-att)/(1-self.r+1e-6) + 1e-6)).mean()
        if torch.any(torch.isnan(info_loss)):
            print('INFO LOSS NAN')
            exit()
        return info_loss

class ConnectLoss_Extended(nn.Module):
    def __init__(self):
        super(ConnectLoss_Extended, self).__init__()

    def forward(self, src, times, smoother_stats, logits):
        #print('logits', logits.shape)
        shift1 = logits[:,1:]
        shift2 = logits[:,:-1]

        # Also normalizes mask
        connect = torch.mean(torch.sqrt((shift1 - shift2) ** 2))
        return connect

class ConnectLoss(nn.Module):
    def __init__(self):
        super(ConnectLoss, self).__init__()

    def forward(self, logits):
        shift1 = logits[:,1:,:]
        shift2 = logits[:,:-1,:]

        # print('shift1', shift1.shape)
        # print('shift2', shift2.shape)
        # print('abs', torch.abs(shift1 - shift2).sum())
        # print('comp 1', torch.sum(torch.abs(shift1 - shift2)).shape)

        # Also normalizes mask
        connect = torch.sum((shift1 - shift2).norm(p=2)) / shift1.flatten().shape[0]

        #print('Connect', connect.shape)

        return connect

class DimEntropy(nn.Module):
    def __init__(self, dim = 2):
        '''
        dim == 2 means we use sensor-wise entropy
        dim == 1 is time-wise entropy
        '''
        super(DimEntropy, self).__init__()
        self.dim = dim

    def forward(self, mask):
        # Flip dimension:
        # print('mask check', mask.shape)
        # exit()
        if len(mask.shape) > 3:
            mask = mask.squeeze(-1)
        sum_dim = 1 if self.dim == 2 else 2
        dist = mask.sum(sum_dim)
        dist /= dist.sum(1).unsqueeze(-1).repeat(1,mask.shape[self.dim])
        ent = -1.0 * (dist.log() * dist).sum(dim=1).mean() # Take mean across batches
        return ent

class PairwiseDecorrelation(nn.Module):
    def __init__(self):
        super(PairwiseDecorrelation, self).__init__()

    def forward(self, mask_list):
        return (mask_list[0] * mask_list[1]).mean() 

class EntropyConceptSimDistribution(nn.Module):
    def __init__(self):
        super(EntropyConceptSimDistribution, self).__init__()
    def forward(self, ze, zc):
        # ze: size (B, d_z)
        # zc: size (Nc, naug, d_z)
        Nc, naug, _ = zc.shape

        ze = F.normalize(ze, dim = -1)
        zc = F.normalize(zc, dim = -1)
        #zc = zcon_dist.flatten(0, 1).repeat(B, 1) # Size (Nc x naug, d_z) -> (B x Nc x naug, d_z)
        ze = ze.unsqueeze(0).repeat(Nc, 1, 1) # Shape (Nc, B, d_z)
        zc = zc.transpose(1, 2) # Shape (Nc, d_z, naug)

        sims = torch.bmm(ze, zc).transpose(0, 1) # Shape (Nc, B, naug) -> (B, Nc, naug)

        # Take mean of distances to all concept embeddings:
        sim_probs = sims.mean(dim = -1).softmax(dim = -1) # Shape (B, Nc) -> (B, Nc)

        # Take entropy loss across sim distribution:
        ent = -1.0 * (torch.log(sim_probs + 1e-9) * sim_probs).sum(dim=-1).mean()

        return ent

class EntropyPrototypeSimDistribution(nn.Module):
    def __init__(self):
        super(EntropyPrototypeSimDistribution, self).__init__()
    def forward(self, ze, zp):
        # ze: size (B, d_z)
        # zp: size (Np, d_z)

        ze = F.normalize(ze, dim = -1)
        zp = F.normalize(zp, dim = -1)

        sims = torch.matmul(ze, zp.transpose(0, 1)) # Shape (B, Np)

        # Take mean of distances to all concept embeddings:
        sim_probs = sims.softmax(dim = -1) # Shape (B, Nc) -> (B, Nc)

        # Take entropy loss across sim distribution:
        ent = -1.0 * (torch.log(sim_probs + 1e-9) * sim_probs).sum(dim=-1).mean()

        return ent