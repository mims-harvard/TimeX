import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def exp_criterion_eval_smoothers(src, times, smoother_stats, beta: float, exp_criterion: torch.nn.Module):

    if not (isinstance(beta, torch.Tensor) and (isinstance(exp_criterion, list))):
        l = exp_criterion(src, times, smoother_stats)
        return beta * l, [l.item()]

    # Else, need list-based evaluation
    llist = []
    for i in range(len(beta)):
        
        l = exp_criterion[i](src, times, smoother_stats)
        llist.append(l.item())

        if i == 0:
            lsum = beta[i] * l
        else:
            lsum += beta[i] * l

    return lsum, llist

class SizeMaskLoss(nn.Module):
    def __init__(self, mean = True):
        super(SizeMaskLoss, self).__init__()
        self.mean = True
    def forward(self, src, times, smoother_stats):
        alpha, beta, thresh, _ = smoother_stats
        if self.mean:
            L = (torch.sin(alpha * times + beta) - thresh).relu().mean()
        else:
            L = (torch.sin(alpha * times + beta) - thresh).relu().sum()

        return L

class PSizeLoss(nn.Module):
    def __init__(self, max_len):
        super(PSizeLoss, self).__init__()  
        self.max_len = max_len
    def forward(self, src, times, smoother_stats):
        _, _, _, p = smoother_stats
        return (-1.0 * p.mean())