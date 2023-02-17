# Decoupled training version of SAT model

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import os
import sys; sys.path.append(os.path.dirname(__file__))
from ..encoders.positional_enc import PositionalEncodingTF

from ..layers import TransformerEncoderInterpret, TransformerEncoderLayerInterpret
from ..extractors import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_extractor(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-1e-10, 1e-10)

class MaskModel(nn.Module):
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    """

    def __init__(self, 
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            enc_dropout = None, # Encoder dropout 
            nhead = 1, # Number of attention heads
            trans_dim_feedforward = 72, # Number of hidden layers 
            trans_dropout = 0.25, # Dropout rate in Transformer encoder
            nlayers = 1, # Number of Transformer layers
            aggreg = 'mean', # Aggregation of transformer embeddings
            MAX = 10000, # Arbitrary large number
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
            d_extractor = 64,
            return_org_attn = False,
            time_mask_only = True,
            use_mlp_encoder = True, # If true, runs input through an encoding before
            no_return_attn = False,
            type_masktoken = 'zero',
            masktoken_kwargs = {},
            seed = None,
            enc_phi_key = 'transformer',
            enc_phi_kwargs = {},
            type_archmask = None,
            **kwargs
            ):

        super(MaskModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_inp = d_inp
        self.max_len = max_len
        self.n_classes = n_classes
        self.enc_dropout = enc_dropout
        self.nhead = nhead
        self.trans_dim_feedforward = trans_dim_feedforward
        self.trans_dropout = trans_dropout
        self.nlayers = nlayers
        self.aggreg = aggreg
        self.static = static
        self.d_static = d_static
        self.d_pe = d_pe

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)
        self.d_inp = d_inp
        self.d_extractor = d_extractor
        self.return_org_attn = return_org_attn

        self.time_mask_only = time_mask_only
        self.use_mlp_encoder = use_mlp_encoder
        self.no_return_attn = no_return_attn

        self.type_archmask = type_archmask

        self.enc_phi_key = enc_phi_key.lower() if enc_phi_key is not None else None
        self.enc_phi_kwargs = enc_phi_kwargs

        if self.enc_phi_key == 'conv':
            self.enc_phi = TransformerConvSNEncPhi(d_inp = d_inp,
                max_len = max_len, d_pe = d_pe, batch_first = False, **self.enc_phi_kwargs)

        elif self.enc_phi_key == 'dualattn':
            self.enc_phi = DATEncPhi(
                max_len = max_len, d_inp = d_inp, d_pe = d_pe, MAX = MAX,
                **self.enc_phi_kwargs,
            )

        else: # All else, set as regular transformer:
            self.enc_phi = TransformerEncPhi(
                d_inp = d_inp, max_len = max_len, d_pe = d_pe, **self.enc_phi_kwargs
            )

        # Set up Transformer encoder layer 1 (to learn extractions):

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if self.use_mlp_encoder:
            self.MLP_encoder = nn.Linear(d_inp, d_inp)
            self.init_weights()

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        self.d_fi = d_fi

        if self.enc_dropout is not None:
            self.enc_dropout = nn.Dropout(dropout)
        else:
            self.enc_dropout = lambda x: x # Identity arbitrary function

        # Make "MASK token", i.e. fixed randn vector
        #self.mask_token = torch.randn(self.max_len).to(device)
        self.type_masktoken = type_masktoken
        self.masktoken_kwargs = masktoken_kwargs
        self.seed = seed
        if self.type_masktoken == 'zero':
            self.mask_token = torch.zeros(self.max_len).to(device)
        elif self.type_masktoken == 'dynamic_normal':
            self.mask_token = None
        elif self.type_masktoken == 'normal':
            self.mask_token = torch.randn(self.max_len).to(device)
        elif self.type_masktoken == 'dyna_norm_datawide':
            mu = self.masktoken_kwargs['mu']
            std = self.masktoken_kwargs['std']
            #torch.manual_seed(self.seed)
            self.mask_token = lambda: mu + torch.randn_like(std) * std
        elif self.type_masktoken == 'norm_datawide':
            mu = self.masktoken_kwargs['mu']
            std = self.masktoken_kwargs['std']
            torch.manual_seed(self.seed)
            self.mask_token = mu + torch.randn_like(std) * std

        elif self.type_masktoken == 'decomp_dyna':
            # Get out decomposition:
            mu_trend = self.masktoken_kwargs['mu_trend']
            std_trend = self.masktoken_kwargs['std_trend']
            mu_seasonal = self.masktoken_kwargs['mu_seasonal']
            std_seasonal = self.masktoken_kwargs['std_seasonal']
            def r(): # Function to return mask replacement features on both trend and seasonal components
                trend_comp = mu_trend + torch.randn_like(std_trend) * std_trend
                seasonal_comp = mu_seasonal + torch.randn_like(std_seasonal) * std_seasonal
                return trend_comp, seasonal_comp
            self.mask_token = r

        elif self.type_masktoken == 'decomp_zero':
            # Get out decomposition:
            def r(): # Function to return mask replacement features on both trend and seasonal components
                trend_comp = torch.zeros(self.max_len, self.d_inp)
                seasonal_comp = torch.zeros(self.max_len, self.d_inp)
                return trend_comp, seasonal_comp
            self.mask_token = r

    def init_weights(self):
        initrange = 1e-10
        self.MLP_encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def generate_mask(self, enc_phi_out):
        '''
        Params:
            enc_phi_out (tensor): outputs of enc_phi encoder
        '''

        raise NotImplementedError('Need to implement generate_mask method')

    def get_to_replace(self, src, times):        
        
        if self.type_masktoken == 'dynamic_normal':
            # Generate new mask:
            # Take sensor-wise mu, std across the sample:
            mu = torch.mean(src, dim=1).unsqueeze(1).repeat(1, src.shape[1], 1)
            std = torch.std(src, dim=1, unbiased = True).unsqueeze(1).repeat(1, src.shape[1], 1)
            to_replace = mu + std * torch.rand_like(src)
        elif self.type_masktoken == 'dyna_norm_datawide':
            to_replace = self.mask_token().unsqueeze(0).repeat(src.shape[0], 1, 1)
            #to_replace = torch.stack([self.mask_token() for _ in range(src.shape[1])], dim = 1)
        elif self.type_masktoken == 'norm_datawide':
            to_replace = self.mask_token.unsqueeze(0).repeat(src.shape[0], 1, 1)
        elif (self.type_masktoken == 'decomp_dyna') or (self.type_masktoken == 'decomp_zero'):
            trend, sea = self.mask_token()
            trend = trend.unsqueeze(0).repeat(src.shape[0], 1, 1)
            sea = sea.unsqueeze(0).repeat(src.shape[0], 1, 1)
            to_replace = (trend, sea)
        else:
            to_replace = self.mask_token.unsqueeze(-1).unsqueeze(0).repeat(src.shape[0], 1, src.shape[-1])

        return to_replace

    def apply_mask(self, src, times, mask, captum_input = False):
        
        # Simple element-wise multiplication for now
        #   Not necessarily in-distribution - will address problem later

        if not captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)
            #mask = mask.transpose(0, 1)

        # Leaves src, times as B,T,d
        to_replace = self.get_to_replace(src, times)

        if len(mask.shape) < 2:
            M = mask.unsqueeze(-1).repeat(1,1,src.shape[-1])
        else:
            M = mask
        
        src = src * M + to_replace * (1 - M)

        #times = times # Gate the times as well

        return src, times


    def enc_theta(self, src, times, joint_mask = None, captum_input = False):
        
        # If training with the full, feature-wise mask, assuming joint_mask is already in the shape
        #   of a full mask on the input values

        if not captum_input:
            # Flip from (T, B, d) -> (B, T, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1)#.transpose(1, 2) # Flip from (T,B) -> (B,T) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=0)

        if joint_mask is None:
            lengths = torch.sum(times > 0, dim=1) # Lengths should be size (B,)
            maxlen, batch_size = src.shape[1], src.shape[0]
            joint_mask = ~(torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])).to(device)
            joint_mask = joint_mask.unsqueeze(-1).repeat(1,1,src.shape[-1])

        if len(joint_mask.shape) < 3 or (joint_mask.shape[-1] == 1):
            joint_mask = joint_mask.unsqueeze(-1).repeat(1,1,src.shape[-1])

        enc2_x = torch.cat([src,joint_mask], dim=2) # MASK ON SRC (ORIGINAL INPUT)
        # enc2_x needs to be (B,T,d), so need the transpose above
        # Mask is applied on the end of enc2_x
        output = self.encoder2(enc2_x, times)

        return output

    def forward(self,
            src,
            times,
            static = None,
            mask = None,
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            ):
        '''
        * Ensure all inputs are cuda before calling forward method

        Dimensions of inputs:
            (B = batch, T = time, d = dimensions of each time point)
            src = (T, B, d)
            times = (T, B)

        Times must be length of longest sample in dataset, with 0's padded at end

        Parameters:
            mask (tensor): Can provide manual mask on which to apply to inputs. Should be size (B,T)
        '''

        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        out_phi, overlen_mask = self.enc_phi(src, times, static = static, captum_input = False)

        if mask is None:
            mask = self.generate_mask(out_phi)

        not_overlen = ~overlen_mask
        if len(mask.shape) > 2:
            #Must repeat overlen mask to match size of mask:
            not_overlen = not_overlen.unsqueeze(-1).repeat(1,1,mask.shape[-1])

        #print('overlen_mask', not_overlen.shape)

        joint_mask = mask * (not_overlen).float() # Combine with over-length mask

        masked_src, masked_times = self.apply_mask(src, times, mask = joint_mask, captum_input = False)

        output = self.enc_theta(masked_src, masked_times, ~overlen_mask, captum_input = True)

        return output, mask