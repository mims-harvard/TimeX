import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from txai.utils.predictors.loss import GSATLoss_Extended, ConnectLoss_Extended

def exp_criterion_eval_smoothers(src, times, mask, smoother_stats, beta: float, exp_criterion: torch.nn.Module):

    if not (isinstance(beta, torch.Tensor) and (isinstance(exp_criterion, list))):
        l = exp_criterion(src, times, smoother_stats, mask)
        return beta * l, [l.item()]

    # Else, need list-based evaluation
    llist = []
    for i in range(len(beta)):
        
        l = exp_criterion[i](src, times, smoother_stats, mask)
        llist.append(l.item())

        if i == 0:
            lsum = beta[i] * l
        else:
            lsum += beta[i] * l

    return lsum, llist

class CurveSizeLoss(nn.Module):
    def __init__(self, mean = True):
        super(CurveSizeLoss, self).__init__()
        self.mean = True
    def forward(self, src, times, smoother_stats, mask):
        alpha, beta, thresh, _ = smoother_stats
        if self.mean:
            L = F.softplus(torch.sin(alpha * times + beta) - thresh).mean()
        else:
            L = F.softplus(torch.sin(alpha * times + beta) - thresh).sum()

        return L

class SizeMaskLoss_SS(nn.Module):
    def __init__(self, mean = True, target_val = None):
        super(SizeMaskLoss, self).__init__()
        self.mean = mean
        self.target_val = target_val
    def forward(self, src, times, smoother_stats, mask):
        if self.mean:
            if self.target_val is not None:
                return torch.sqrt((mask.mean() - self.target_val) ** 2)
            else:
                return mask.mean()
        else:
            if self.target_val is not None:
                #print((mask.sum() / mask.shape[0]))
                return torch.sqrt(((mask.sum() / mask.shape[0]) - self.target_val) ** 2)
            else:
                return mask.sum() / mask.shape[0]

class SizeMaskLoss(nn.Module):
    def __init__(self, mean = True, target_val = None):
        super(SizeMaskLoss, self).__init__()
        self.mean = mean
        self.target_val = target_val
    def forward(self, mask):
        if self.mean:
            if self.target_val is not None:
                return (mask.mean() - self.target_val).norm(p=2)
            else:
                return mask.mean()
        else:
            if self.target_val is not None:
                #print((mask.sum() / mask.shape[0]))
                return ((mask.sum() / mask.shape[0]) - self.target_val).norm(p=2)
            else:
                return mask.sum() / mask.shape[0]

class EMaskLoss(nn.Module):
    def __init__(self, target_val = 5):
        super(EMaskLoss, self).__init__()
        self.target_val = 5
    def forward(self, d_1, d_2):
        # ref = torch.arange(d_1.shape[1], dtype = torch.float32, requires_grad = False).to(d_1.device).unsqueeze(1)
        # Ed_1 = torch.matmul(d_1, ref).squeeze()
        # Ed_2 = torch.matmul(d_2, ref).squeeze()

        #return (Ed_1 - Ed_2).norm(p=2)

        return (d_1 - d_2).norm(p=2)

class CDFMaskLoss(nn.Module):
    def __init__(self):
        super(CDFMaskLoss, self).__init__()
    def forward(self, d_1, d_2):
        # ref = torch.arange(d_1.shape[1], dtype = torch.float32, requires_grad = False).to(d_1.device).unsqueeze(1)
        # Ed_1 = torch.matmul(d_1, ref).squeeze()
        # Ed_2 = torch.matmul(d_2, ref).squeeze()

        #return (Ed_1 - Ed_2).norm(p=2)

        # Get CDF:
        B, T = d_1.shape
        assert B == d_2.shape[0]

        #lT = torch.tril(torch.ones(T, T)).unsqueeze(0).repeat(B,1,1)
        L_n = torch.triu(torch.ones(T, T)).to(d_1.device)

        cdf_1 = torch.matmul(d_1, L_n) # (B, T) x (T, T)
        cdf_2 = torch.matmul(d_2, L_n) # (B, T) x (T, T)

        return (cdf_1 - cdf_2).norm(p=2, dim = -1).mean()

class KLDMaskLoss(nn.Module):
    def __init__(self):
        super(KLDMaskLoss, self).__init__()
    def forward(self, d_1, d_2):
        return -1.0 * torch.nn.functional.kl_div(d_1, d_2).mean()

class PSizeLoss_SS(nn.Module):
    def __init__(self, max_len, margin = 0.9):
        super(PSizeLoss, self).__init__()  
        self.max_len = max_len
        self.margin = margin
    def forward(self, src, times, smoother_stats, mask):
        p = (self.margin - smoother_stats).relu()
        return (p.mean())

class PSizeLoss(nn.Module):
    def __init__(self, margin = 0.1):
        super(PSizeLoss, self).__init__()  
        self.margin = margin
    def forward(self, p):
        p_margin = (p - self.margin).relu()
        return p_margin.mean()

class ProbSmoothLoss(nn.Module):
    def __init__(self):
        super(ProbSmoothLoss, self).__init__()
    def forward(self, logits, smoothed_src):
        smoothed_src1 = smoothed_src[1:,:,:]
        smoothed_src2 = smoothed_src[:-1,:,:]
        probs = (logits[:,1:] * logits[:,:-1]) + torch.sqrt((smoothed_src1 - smoothed_src2) ** 2)

        if torch.any(torch.isnan(probs)):
            raise ValueError("Probs has NaNs")

        return probs.mean() 


class InterpretabilityCriterion(nn.Module):
    def __init__(self, r, lam = 1.0):
        super(InterpretabilityCriterion, self).__init__()
        self.gsat_loss = GSATLoss_Extended(r = r)
        self.connect_mask_loss = ConnectLoss_Extended()
        self.smooth_loss = ProbSmoothLoss()
        self.lam = lam

    def forward(self, src, times, out_dict):
        smoother_stats = out_dict['smoother_stats']
        mask = out_dict['mask_logits']
        smooth_src = out_dict['smooth_src']
        print('smooth', smooth_src.isnan().sum()) 
        gsat = self.gsat_loss(src, times, smoother_stats, mask)
        mask_connected = self.connect_mask_loss(src, times, smoother_stats, mask)
        smooth = self.smooth_loss(mask, smooth_src)
        return gsat + mask_connected + self.lam * smooth


