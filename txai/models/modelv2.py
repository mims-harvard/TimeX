import torch
import math
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother
from txai.models.mask_generators.gumbelmask_model import STENegInf
from txai.utils.functional import transform_to_attn_mask

class STEThreshold(torch.autograd.Function):
    # Credit: https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class STENegInfMod(torch.autograd.Function):
    # From: https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    @staticmethod
    def forward(ctx, input):
        #print('input', input.sum())
        mfill = torch.zeros_like(input).masked_fill(input < 1e-9, -1e9).to(input.device)
        # Collapse down to sequence-level:
        #mfill = mfill.sum(-1)
        return mfill

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class CycleTrendGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            alpha_net_d_z = 32,
            beta_net_d_z = 32,
            thresh_net_d_z = 32,
            trend_net_d_z = 32,
        ):
        super(CycleTrendGenerator, self).__init__()

        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.alpha_net_d_z = alpha_net_d_z
        self.beta_net_d_z = beta_net_d_z
        self.thresh_net_d_z = thresh_net_d_z
        self.trend_net_d_z = trend_net_d_z
        self.agg = agg
        self.max_len = max_len
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.alpha_net_d_z),
            nn.PReLU(),
            nn.Linear(self.alpha_net_d_z, 1),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.beta_net_d_z),
            nn.PReLU(),
            nn.Linear(self.beta_net_d_z, 1),
        )

        self.thresh_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, 1),
        )

        self.trend_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.trend_net_d_z),
            nn.PReLU(),
            nn.Linear(self.trend_net_d_z, 1),
        )

    def mask_in_sine_curve(self, times, alpha, beta, thresh):
        mask_in = (torch.sin(alpha * times.transpose(0, 1) + beta) - thresh).relu()

        # Mask in all above zero (i.e. above threshold)

        return mask_in

    def forward(self, z_seq, src, times):

        z_pre_agg = self.pre_agg_net(z_seq)
        
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        alpha = self.alpha_net(agg_z).relu() + (2 * math.pi / self.max_len) # Should change parameterization later
        beta = self.beta_net(agg_z).tanh() * math.pi # Map to [-pi, pi]
        thresh = self.thresh_net(agg_z).tanh() # Map to [-1,1]

        mask_in = self.mask_in_sine_curve(times, alpha, beta, thresh) # Masks-in values

        p = self.trend_net(agg_z).sigmoid() * self.max_len

        ste_mask = STEThreshold.apply(mask_in)
        # print('src', src.shape)
        #print('p', p.shape)
        # Transpose both src and times below bc expecting batch-first input
        smooth_src = smoother(src, times, p, mask = ste_mask)

        return smooth_src, mask_in, (alpha, beta, thresh, p)


class Modelv2(nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_extraction_blocks = 1,
            mask_seed = None,
            enc_dropout = None, # Encoder dropout 
            nhead = 1, # Number of attention heads
            trans_dim_feedforward = 72, # Number of hidden layers 
            trans_dropout = 0.25, # Dropout rate in Transformer encoder
            nlayers = 1, # Number of Transformer layers
            aggreg = 'mean', # Aggregation of transformer embeddings
            norm_embedding = True,
            MAX = 10000, # Arbitrary large number
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
        ):
        super(Modelv2, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes
        self.n_extraction_blocks = n_extraction_blocks
        
        self.encoder = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            enc_dropout = enc_dropout, # Encoder dropout 
            nhead = nhead, # Number of attention heads
            trans_dim_feedforward = trans_dim_feedforward, # Number of hidden layers 
            trans_dropout = trans_dropout, # Dropout rate in Transformer encoder
            nlayers = nlayers, # Number of Transformer layers
            aggreg = aggreg, # Aggregation of transformer embeddings
            MAX = MAX, # Arbitrary large number
            static=static, # Whether to use some static vector in additional to time-varying
            d_static = d_static, # Dimensions of static input  
            d_pe = d_pe, # Dimension of positional encoder
            norm_embedding = norm_embedding
        )

        self.mask_generators = nn.ModuleList([
            CycleTrendGenerator(d_z = d_inp + d_pe, max_len = max_len),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        z_seq = self.encoder.embed(src, times, aggregate = False, captum_input = False)

        smooth_src, mask_in, smoother_stats = self.mask_generators[0](z_seq, src, times)
        
        #print('mask_in', mask_in.sum())

        #ste_mask = STENegInfMod.apply(mask_in)
        ste_mask = STEThreshold.apply(mask_in)

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred = self.encoder(smooth_src, times, attn_mask = ste_mask)

        return pred, mask_in, smoother_stats, smooth_src

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder.n_classes,
            'enc_dropout': self.encoder.enc_dropout,
            'nhead': self.encoder.nhead,
            'trans_dim_feedforward': self.encoder.trans_dim_feedforward,
            'trans_dropout': self.encoder.trans_dropout,
            'nlayers': self.encoder.nlayers,
            'aggreg': self.encoder.aggreg,
            'static': self.encoder.static,
            'd_static': self.encoder.d_static,
            'd_pe': self.encoder.d_pe,
            'norm_embedding': self.encoder.norm_embedding,
            'n_extraction_blocks': self.n_extraction_blocks,
        }