import torch
import math
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother
from txai.models.mask_generators.gumbelmask_model import STENegInf
from txai.utils.functional import transform_to_attn_mask
from txai.models.modelv2 import MaskGenStochastic

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

class Modelv3(nn.Module):
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
            trend_smoother = True
        ):
        super(Modelv3, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes
        self.n_extraction_blocks = n_extraction_blocks
        self.trend_smoother = trend_smoother
        
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
            MaskGenStochastic(d_z = d_inp + d_pe, max_len = max_len, trend_smoother = trend_smoother),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_seq = self.encoder(src, times, captum_input = False, get_embedding = True)

        # Generate smooth_src:
        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times)

        # Feed smooth_src to encoder:

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred_tilde = self.encoder(smooth_src, times, attn_mask = ste_mask, get_embedding = False)

        return pred_regular, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src

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
            'trend_smoother': self.trend_smoother,
        }

class Modelv3SepExtractor(nn.Module):
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
            trend_smoother = True
        ):
        super(Modelv3SepExtractor, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes
        self.n_extraction_blocks = n_extraction_blocks
        self.trend_smoother = trend_smoother
        
        self.encoder_main = TransformerMVTS(
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
        self.encoder_side = TransformerMVTS(
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
            MaskGenStochastic(d_z = (d_inp + d_pe) * 2, max_len = max_len, trend_smoother = trend_smoother),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_seq_main = self.encoder_main(src, times, captum_input = False, get_embedding = True)
        z_seq_side = self.encoder_side.embed(src, times, captum_input = False, aggregate = False)
        z_seq = torch.cat([z_seq_main, z_seq_side], dim=-1)

        # Generate smooth_src:
        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times)

        # Feed smooth_src to encoder:

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred_tilde = self.encoder_main(smooth_src, times, attn_mask = ste_mask, get_embedding = False)

        return pred_regular, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'enc_dropout': self.encoder_main.enc_dropout,
            'nhead': self.encoder_main.nhead,
            'trans_dim_feedforward': self.encoder_main.trans_dim_feedforward,
            'trans_dropout': self.encoder_main.trans_dropout,
            'nlayers': self.encoder_main.nlayers,
            'aggreg': self.encoder_main.aggreg,
            'static': self.encoder_main.static,
            'd_static': self.encoder_main.d_static,
            'd_pe': self.encoder_main.d_pe,
            'norm_embedding': self.encoder_main.norm_embedding,
            'n_extraction_blocks': self.n_extraction_blocks,
            'trend_smoother': self.trend_smoother,
        }

class Modelv3EmbSim(nn.Module):
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
            trend_smoother = True
        ):
        super(Modelv3EmbSim, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes
        self.n_extraction_blocks = n_extraction_blocks
        self.trend_smoother = trend_smoother
        
        self.encoder_main = TransformerMVTS(
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
        self.encoder_side = TransformerMVTS(
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
            MaskGenStochastic(d_z = (d_inp + d_pe) * 2, max_len = max_len, trend_smoother = trend_smoother),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_seq_main = self.encoder_main(src, times, captum_input = False, get_embedding = True)
        z_seq_side = self.encoder_side.embed(src, times, captum_input = False, aggregate = False)
        z_seq = torch.cat([z_seq_main, z_seq_side], dim=-1)

        # Generate smooth_src:
        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times)

        # Feed smooth_src to encoder:

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred_tilde, z_seq_tilde = self.encoder_main(smooth_src, times, attn_mask = ste_mask, get_embedding = True)

        return pred_regular, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src, (z_seq_main, z_seq_tilde)

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'enc_dropout': self.encoder_main.enc_dropout,
            'nhead': self.encoder_main.nhead,
            'trans_dim_feedforward': self.encoder_main.trans_dim_feedforward,
            'trans_dropout': self.encoder_main.trans_dropout,
            'nlayers': self.encoder_main.nlayers,
            'aggreg': self.encoder_main.aggreg,
            'static': self.encoder_main.static,
            'd_static': self.encoder_main.d_static,
            'd_pe': self.encoder_main.d_pe,
            'norm_embedding': self.encoder_main.norm_embedding,
            'n_extraction_blocks': self.n_extraction_blocks,
            'trend_smoother': self.trend_smoother,
        }