import torch
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother, exponential_smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.modelv2 import MaskGenStochasticDecoder
from txai.models.mask_generators.window_mask import MaskGenWindow

transformer_default_args = {
    'enc_dropout': None,
    'nhead': 1,
    'trans_dim_feedforward': 72,
    'trans_dropout': 0.25,
    'nlayers': 1,
    'aggreg': 'mean',
    'MAX': 10000,
    'static': False,
    'd_static': 0,
    'd_pe': 16,
    'norm_embedding': True,
}


class Modelv5Univariate(nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            transformer_args = transformer_default_args,
            trend_smoother = True
        ):
        super(Modelv5Univariate, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.trend_smoother = trend_smoother
        self.transformer_args = transformer_args
        
        self.encoder_main = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args
        )
        self.encoder_pret = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )
        self.encoder_t = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )

        self.mask_generators = nn.ModuleList([
            MaskGenStochasticDecoder(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_main, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        # Generate smooth_src:
        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times, get_tilde_mask = False)

        # Feed smooth_src to encoder:

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred_mask, z_mask, z_seq_mask = self.encoder_t(smooth_src, times, attn_mask = ste_mask, get_agg_embed = True)

        # Return is dictionary:
        d = {
            'pred': pred_regular,
            'pred_mask': pred_mask,
            'mask_logits': mask_in,
            'ste_mask': ste_mask,
            'smoother_stats': smoother_stats, 
            'smooth_src': smooth_src,
            'all_preds': (pred_regular, pred_mask),
            'all_z': (z_main, z_mask)
        }

        return d

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'transformer_args': self.transformer_args,
            'trend_smoother': self.trend_smoother,
        }

class Modelv5Univariate_Window(nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            transformer_args = transformer_default_args,
            trend_smoother = True
        ):
        super(Modelv5Univariate_Window, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.trend_smoother = trend_smoother
        self.transformer_args = transformer_args
        
        self.encoder_main = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args
        )
        self.encoder_pret = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )
        self.encoder_t = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )

        self.mask_generators = nn.ModuleList([
            MaskGenWindow(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother),
        ])

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_main, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        # Generate smooth_src:
        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times, get_tilde_mask = False)

        # Feed smooth_src to encoder:

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred_mask, z_mask, z_seq_mask = self.encoder_t(smooth_src, times, attn_mask = ste_mask, get_agg_embed = True)

        # Return is dictionary:
        d = {
            'pred': pred_regular,
            'pred_mask': pred_mask,
            'mask_logits': mask_in,
            'ste_mask': ste_mask,
            'smoother_stats': smoother_stats, 
            'smooth_src': smooth_src,
            'all_preds': (pred_regular, pred_mask),
            'all_z': (z_main, z_mask)
        }

        return d

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'transformer_args': self.transformer_args,
            'trend_smoother': self.trend_smoother,
        }