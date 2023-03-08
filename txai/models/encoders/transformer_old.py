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
        if self.norm_embedding:
            lnorm = nn.LayerNorm(self.d_pe + d_inp) # self.d_pe + d_inp
            self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers, norm = lnorm)
        else:
            self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers)

        # self.transformer_encoder = Reformer(
        #     dim = self.d_pe + d_inp,
        #     depth = self.nlayers,
        #     heads = self.nhead,
        #     lsh_dropout = 0.1,
        #     causal = False,
        #     pkm_num_keys = 64,
        #     ff_chunks = 400,
        #     #emb_dim = 64,
        #     attn_chunks = 16,
        #     bucket_size = 50,
        # )

        # Encode input
        self.MLP_encoder = nn.Linear(d_inp, d_inp)

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        # Classification head
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.Linear(d_fi, n_classes),
        )

        self.relu = nn.ReLU()

        if self.enc_dropout is not None:
            self.enc_dropout_layer = nn.Dropout(dropout)
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
            given_time_mask = None,
            given_attn_mask = None,
            mask = None,
            aggregate = True,
        ):
        #print('src at entry', src.isnan().sum())

        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        if show_sizes:
            print('captum input = {}'.format(captum_input), src.shape, 'time:', times.shape)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if show_sizes:
            print('torch.sum(times > 0, dim=0)', lengths.shape)

        # Encode input vectors
        #src = self.MLP_encoder(src)

        if show_sizes:
            print('self.MLP_encoder(src)', src.shape)

        # Must flip times to (T, B) for positional encoder
        if src.isnan().sum() > 0:
            print('src before pe', src.isnan().sum())
        pe = self.pos_encoder(times) # Positional encoder
        #pe = times.unsqueeze(-1)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if pe.isnan().sum() > 0:
            print('pe', pe.isnan().sum())
        if src.isnan().sum() > 0:
            print('src after pe', src.isnan().sum())

        if show_sizes:
            print('torch.cat([pe, src], axis=2)', x.shape)

        if self.enc_dropout is not None:
            x = self.enc_dropout_layer(x)

        if show_sizes:
            print('self.enc_dropout(x)', x.shape)

        if static is not None:
            emb = self.emb(static)

        # mask out the all-zero rows
        mask = (torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])).cuda()
        #print('org mash', mask)

        if self.time_rand_mask_size is not None:
            # Calculate random time mask
            msize = int(self.time_rand_mask_size * x.shape[0]) \
                if (self.time_rand_mask_size < 1) else int(self.time_rand_mask_size)

            given_time_mask = torch.ones((x.shape[1], x.shape[0]), dtype=bool).to(mask.get_device())
            for i in range(x.shape[1]):
                rand_inds = torch.randperm(x.shape[0])[:msize]
                given_time_mask[i,rand_inds] = False

            mask = mask | given_time_mask # Combine mask and given mask

        if self.attn_rand_mask_size is not None:
            # Calculate random attention mask
            msize = int(self.attn_rand_mask_size * (x.shape[0] ** 2)) \
                if (self.attn_rand_mask_size < 1) else int(self.attn_rand_mask_size)
            
            # TODO: make robust for multi-head dimension
            given_attn_mask = torch.zeros((x.shape[1], x.shape[0], x.shape[0]), dtype=bool).to(x.get_device())
            rand_i = torch.randint(low = 0, high = x.shape[0], size=(msize,))
            rand_j = torch.randint(low = 0, high = x.shape[0], size=(msize,))

            given_attn_mask[rand_i,rand_j] = True

        if mask.dim() == 1:
            # Unsqueeze if only using one example (must have B first in dim.)
            mask = mask.unsqueeze(dim=0)

        if show_sizes:
            print('mask', mask.shape)

        # Transformer must have (T, B, d)
        # src_key_padding_mask is (B, T)
        # mask is (B*n_heads,T,T) - if None has no effect
        if x.isnan().sum() > 0:
            print('before enc', x.isnan().sum())
        output, attn = self.transformer_encoder(x, mask = given_attn_mask, src_key_padding_mask=mask)

        if show_sizes:
            print('transformer_encoder', output.shape)

        # Aggregation scheme:
        if aggregate:
            # Transformer embeddings through MLP --------------------------------------
            mask2 = mask.permute(1, 0).unsqueeze(2).long()

            if show_sizes:
                print('mask.permute(1, 0).unsqueeze(2).long()', mask2.shape)

            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
            elif self.aggreg == 'max':
                output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

            if show_sizes:
                print('self.aggreg: {}'.format(self.aggreg), output.shape)

            if static is not None: # Use embedding of static vector:
                output = torch.cat([output, emb], dim=1)

        # TODO: static if aggregate is False


        return output

    def forward(self, 
            src, 
            times, 
            static = None, 
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            show_sizes = False, # Used for debugging
            given_time_mask = None,
            given_attn_mask = None,
            mask = None
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

        out = self.embed(src, times,
            static = static,
            captum_input = captum_input,
            show_sizes = show_sizes,
            given_time_mask = given_time_mask,
            given_attn_mask = given_attn_mask,
            mask = mask)

        output = self.mlp(out)

        if show_sizes:
            print('self.mlp(output)', output.shape)

        if self.no_return_attn:
            return output
        return output 

class TransformerMVTS_CLFToken(nn.Module):
    """ Transformer model with context embedding, split dimension positional and element embedding

    NOTE: Different from TransformerMVTS in that it uses a CLF Token

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
            MAX = 10000, # Arbitrary large number
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
            time_rand_mask_size = None,
            attn_rand_mask_size = None,
            no_return_attn = False,
            ):

        super(TransformerMVTS_CLFToken, self).__init__()
        self.model_type = 'Transformer'
        self.d_inp = d_inp
        self.max_len = max_len
        self.n_classes = n_classes
        self.enc_dropout = enc_dropout
        self.nhead = nhead
        self.trans_dim_feedforward = trans_dim_feedforward
        self.trans_dropout = trans_dropout
        self.nlayers = nlayers
        self.static = static
        self.d_static = d_static
        self.d_pe = d_pe

        self.time_rand_mask_size = time_rand_mask_size
        self.attn_rand_mask_size = attn_rand_mask_size
        self.no_return_attn = no_return_attn

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        #Set up Transformer encoder:
        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + d_inp, 
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers)

        # self.transformer_encoder = Reformer(
        #     dim = self.d_pe + d_inp,
        #     depth = self.nlayers,
        #     heads = self.nhead,
        #     lsh_dropout = 0.1,
        #     causal = False,
        #     pkm_num_keys = 64,
        #     ff_chunks = 400,
        #     #emb_dim = 64,
        #     attn_chunks = 16,
        #     bucket_size = 50,
        # )

        # Encode input
        self.MLP_encoder = nn.Linear(d_inp, d_inp)

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        # Classification head
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.Linear(d_fi, n_classes),
        )

        self.relu = nn.ReLU()

        if self.enc_dropout is not None:
            self.enc_dropout = nn.Dropout(dropout)
        else:
            self.enc_dropout = lambda x: x # Identity arbitrary function

        # Initialize weights of module
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.MLP_encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, 
            src, 
            times, 
            static = None, 
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            show_sizes = False, # Used for debugging
            given_time_mask = None,
            given_attn_mask = None,
            mask = None
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

        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1)#.transpose(1, 2) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            # if captum_input:
            #     src = src.unsqueeze(dim=0)
            # else:
            src = src.unsqueeze(dim=1)

        if show_sizes:
            print('captum input = {}'.format(captum_input), src.shape, 'time:', times.shape)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if show_sizes:
            print('torch.sum(times > 0, dim=0)', lengths.shape)

        #assert src.shape[2] == self.d_inp, f'Source shape does not match d_inp (src={src.shape[2]} vs. d_inp={self.d_inp})'

        # Encode input vectors
        #src = self.MLP_encoder(src)

        if show_sizes:
            print('self.MLP_encoder(src)', src.shape)

        # Must flip times to (T, B) for positional encoder
        pe = self.pos_encoder(times) # Positional encoder
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if show_sizes:
            print('torch.cat([pe, src], axis=2)', x.shape)

        x = self.enc_dropout(x)

        if show_sizes:
            print('self.enc_dropout(x)', x.shape)

        if static is not None:
            emb = self.emb(static)

        # mask out the all-zero rows
        mask = (torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])).cuda()
        #print('org mash', mask)

        if self.time_rand_mask_size is not None:
            # Calculate random time mask
            msize = int(self.time_rand_mask_size * x.shape[0]) \
                if (self.time_rand_mask_size < 1) else int(self.time_rand_mask_size)

            #print('msize', msize)

            given_time_mask = torch.ones((x.shape[1], x.shape[0]), dtype=bool).to(mask.get_device())
            for i in range(x.shape[1]):
                rand_inds = torch.randperm(x.shape[0])[:msize]
                #print(rand_inds)
                given_time_mask[i,rand_inds] = False

            #print('given_time_mask > 1', (given_time_mask[0] > 0).sum().item())
            #exit()

            # print('mask size', mask.shape)
            # print('given_time_mask', given_time_mask.shape)
            # exit()

            mask = mask | given_time_mask # Combine mask and given mask

            #print('mask', mask.sum().item() / mask.flatten().shape[0])
            # print('mask shape', mask.shape)

        if self.attn_rand_mask_size is not None:
            # Calculate random attention mask
            msize = int(self.attn_rand_mask_size * (x.shape[0] ** 2)) \
                if (self.attn_rand_mask_size < 1) else int(self.attn_rand_mask_size)
            
            # TODO: make robust for multi-head dimension
            given_attn_mask = torch.zeros((x.shape[1], x.shape[0], x.shape[0]), dtype=bool).to(x.get_device())
            rand_i = torch.randint(low = 0, high = x.shape[0], size=(msize,))
            rand_j = torch.randint(low = 0, high = x.shape[0], size=(msize,))

            given_attn_mask[rand_i,rand_j] = True

        if mask.dim() == 1:
            # Unsqueeze if only using one example (must have B first in dim.)
            mask = mask.unsqueeze(dim=0)

        # Add CLF token to the first index in x
        clf_token = torch.zeros(1, x.shape[1], x.shape[2]).float().to(x.get_device())
        clf_token[-1] = -1
        x = torch.cat([clf_token, x], dim = 0)

        # Add additional value to mask
        newmask = torch.zeros(mask.shape[0], mask.shape[1] + 1)
        newmask[:,1:] = mask
        mask = newmask.cuda()

        if show_sizes:
            print('x after clf', x.shape)
            print('mask', mask.shape)

        # Transformer must have (T, B, d)
        # src_key_padding_mask is (B, T)
        # mask is (B*n_heads,T,T) - if None has no effect
        output, attn = self.transformer_encoder(x, mask = given_attn_mask, src_key_padding_mask=mask)
        #print('x shape', x.transpose(0,1).shape)
        #output = self.transformer_encoder(x.transpose(0,1)).transpose(0,1)

        if show_sizes:
            print('transformer_encoder', output.shape)

        # Transformer embeddings through MLP --------------------------------------
        mask2 = mask.permute(1, 0).unsqueeze(2).long()

        if show_sizes:
            print('mask.permute(1, 0).unsqueeze(2).long()', mask2.shape)

        # Aggregation scheme:
        # if self.aggreg == 'mean':
        #     lengths2 = lengths.unsqueeze(1)
        #     output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        # elif self.aggreg == 'max':
        #     output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        output = output[0,:,:] # Reduce to clf token (0th index)

        # Feed through MLP:
        if static is not None: # Use embedding of static vector:
            output = torch.cat([output, emb], dim=1)

        output = self.mlp(output)

        if show_sizes:
            print('self.mlp(output)', output.shape)

        if self.no_return_attn:
            return output
        return output, None#, attn