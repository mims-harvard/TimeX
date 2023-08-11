import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from reformer_pytorch import Reformer

import os, ipdb
import sys; sys.path.append(os.path.dirname(__file__))
from .positional_enc import PositionalEncodingTF
from ..layers import TransformerEncoderInterpret, TransformerEncoderLayerInterpret
#from torch.nn import TransformerEncoder, TransformerEncoderLayer

pam_config = {
    'd_inp': 17,
    'd_model': 36,
    'nhead': 1,
    'nhid': 2 * 36,
    'nlayers': 1,
    'dropout': 0.3,
    'max_len': 600,
    'd_static': 0,
    'MAX': 100,
    'aggreg': 'mean',
    'n_classes': 8,
    'perc':  0.5,
    'static': False,
}

class TransformerMVTS(nn.Module):
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
            norm_embedding = False,
            time_rand_mask_size = None,
            attn_rand_mask_size = None,
            no_return_attn = True,
            pre_seq_mlp = False,
            stronger_clf_head = False,
            pre_agg_transform = False,
            ):

        super(TransformerMVTS, self).__init__()
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
        self.norm_embedding = norm_embedding
        self.pre_seq_mlp = pre_seq_mlp
        self.stronger_clf_head = stronger_clf_head
        self.pre_agg_transform = pre_agg_transform

        self.time_rand_mask_size = time_rand_mask_size
        self.attn_rand_mask_size = attn_rand_mask_size
        self.no_return_attn = no_return_attn

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        #Set up Transformer encoder:
        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + d_inp, #self.d_pe + d_inp
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        #if self.norm_embedding:
            #lnorm = nn.LayerNorm(self.d_pe + d_inp) # self.d_pe + d_inp
            #self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers, norm = lnorm)
        #else:
        self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers)

        # Encode input
        self.MLP_encoder = nn.Linear(d_inp, d_inp)

        if self.pre_seq_mlp:
            self.pre_MLP_encoder = nn.Sequential(
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
            )

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        # Classification head
        if stronger_clf_head:
            self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, n_classes),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.ReLU(),
                nn.Linear(d_fi, n_classes),
            )

        if self.pre_agg_transform:
            self.pre_agg_net = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
            )

        self.relu = nn.ReLU()

        if self.enc_dropout is not None:
            self.enc_dropout_layer = nn.Dropout(self.enc_dropout)
        else:
            self.enc_dropout_layer = lambda x: x # Identity arbitrary function

        # Initialize weights of module
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.MLP_encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def set_config(self):
        self.config = {
            'd_inp': self.d_inp,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'enc_dropout': self.enc_dropout,
            'nhead': self.nhead,
            'trans_dim_feedforward': self.trans_dim_feedforward,
            'trans_dropout': self.trans_dropout,
            'nlayers': self.nlayers,
            'aggreg': self.aggreg,
            'static': self.static,
            'd_static': self.d_static,
            'd_pe': self.d_pe,
            'norm_embedding': self.norm_embedding,
        }

    def embed(self, src, times, static = None, captum_input = False,
            show_sizes = False,
            src_mask = None,
            attn_mask = None,
            aggregate = True,
            get_both_agg_full = False,
        ):
        #print('src at entry', src.isnan().sum())

        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        if (src_mask is None) and torch.any(times < -1e5) and (attn_mask is None):
            src_mask = (times < -1e5).transpose(0,1)
            # if attn_mask is not None:
            #     attn_mask *= src_mask.unsqueeze(-1).repeat(1, 1, attn_mask.shape[-1])
            #     src_mask = None

        if show_sizes:
            print('captum input = {}'.format(captum_input), src.shape, 'time:', times.shape)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if show_sizes:
            print('torch.sum(times > 0, dim=0)', lengths.shape)

        # Encode input vectors
        #src = self.MLP_encoder(src)

        if self.pre_seq_mlp:
            src = self.pre_MLP_encoder(src)

        if show_sizes:
            print('self.MLP_encoder(src)', src.shape)

        # Must flip times to (T, B) for positional encoder
        # if src.detach().clone().isnan().sum() > 0:
        #     print('src before pe', src.isnan().sum())
        pe = self.pos_encoder(times) # Positional encoder
        pe = pe.to(src.device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if pe.isnan().sum() > 0:
            print('pe', pe.isnan().sum())
        if src.detach().clone().isnan().sum() > 0:
            print('src after pe', src.isnan().sum())

        if show_sizes:
            print('torch.cat([pe, src], axis=2)', x.shape)

        if self.enc_dropout is not None:
            x = self.enc_dropout_layer(x)

        if show_sizes:
            print('self.enc_dropout(x)', x.shape)

        if static is not None:
            emb = self.emb(static)

        # Transformer must have (T, B, d)
        # src_key_padding_mask is (B, T)
        # mask is (B*n_heads,T,T) - if None has no effect
        if x.isnan().sum() > 0:
            print('before enc', x.isnan().sum())
        output_preagg, attn = self.transformer_encoder(x, src_key_padding_mask = src_mask, mask = attn_mask)

        if show_sizes:
            print('transformer_encoder', output.shape)

        if self.pre_agg_transform:
            output_preagg = self.pre_agg_net(output_preagg)

        # Aggregation scheme:
        if aggregate:
            # Transformer embeddings through MLP --------------------------------------
            #mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if show_sizes:
                print('mask.permute(1, 0).unsqueeze(2).long()', mask2.shape)

            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                if src_mask is not None:
                    #import ipdb; ipdb.set_trace()
                    output = torch.sum(output_preagg * (1 - src_mask.transpose(0,1).unsqueeze(-1).repeat(1, 1, output_preagg.shape[-1]).float()), dim=0) / (lengths2 + 1)
                else:
                    output = torch.sum(output_preagg, dim=0) / (lengths2 + 1)
            elif self.aggreg == 'max':
                output, _ = torch.max(output_preagg, dim=0)

            if show_sizes:
                print('self.aggreg: {}'.format(self.aggreg), output.shape)

            if static is not None: # Use embedding of static vector:
                output = torch.cat([output, emb], dim=1)

        if self.norm_embedding and aggregate:
            output = F.normalize(output, dim = -1)

        if get_both_agg_full:
            return output, output_preagg

        if aggregate:
            return output
        else:
            return output_preagg

    def forward(self, 
            src, 
            times, 
            static = None, 
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            show_sizes = False, # Used for debugging
            attn_mask = None,
            src_mask = None,
            get_embedding = False,
            get_agg_embed = False,
            ):
        '''
        * Ensure all inputs are cuda before calling forward method

        Dimensions of inputs:
            (B = batch, T = time, d = dimensions of each time point)
            src = (T, B, d)
            times = (T, B)

        Times must be length of longest sample in dataset, with 0's padded at end

        Params:
            given_time_mask (torch.Tensor): Mask on which to apply before feeding input into transformer encoder
                - Can provide random mask for baseline purposes
            given_attn_mask (torch.Tensor): Mask on which to apply to the attention mechanism
                - Can provide random mask for baseline comparison
        '''

        #print('src_mask', src_mask.shape)

        out, out_full = self.embed(src, times,
            static = static,
            captum_input = captum_input,
            show_sizes = show_sizes,
            attn_mask = attn_mask,
            src_mask = src_mask,
            get_both_agg_full = True)

        output = self.mlp(out)

        if show_sizes:
            print('self.mlp(output)', output.shape)

        # if self.no_return_attn:
        #     return output
        if get_embedding:
            return output, out_full
        elif get_agg_embed:
            return output, out, out_full
        else:
            return output 