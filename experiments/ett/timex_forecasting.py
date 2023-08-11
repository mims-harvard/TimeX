import torch
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.functional import transform_to_attn_mask
from txai.models.mask_generators.maskgen import MaskGenerator

from txai.utils.predictors.loss import GSATLoss, ConnectLoss
from txai.utils.predictors.loss_smoother_stats import *
from txai.utils.functional import js_divergence, stratified_sample
from txai.models.encoders.simple import CNN, LSTM
from txai.models.bc_model import default_loss_weights

from model import EncoderLayer, DecoderLayer, Transformer

class MaskGeneratorForecasting(torch.nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.d_z = d_z
        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, nhead = 1, dim_feedforward = 32)
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = 2)
        self.time_prob_net = nn.Linear(d_z, 2)

    def forward(self, z_seq, src):
        
        z_seq_dec = self.mask_decoder(tgt = src, memory = z_seq)

        p_time = self.time_prob_net(z_seq_dec)
        total_mask_reparameterize = self.reparameterize(p_time)
        total_mask = p_time.softmax(dim=-1)[...,1]

        return total_mask, total_mask_reparameterize

    def reparameterize(self, total_mask):
        total_mask_log_prob = F.log_softmax(total_mask, dim=-1)
        total_mask_reparameterize = F.gumbel_softmax(total_mask_log_prob, tau = 1.0, hard = True)[...,1]
        return total_mask_reparameterize

class TimeXForecasting(torch.nn.Module):
    def __init__(self, 
            transformer_config, 
            masktoken_stats = None, 
            pooling_method = 'max', 
            r = 0.5,
            loss_weight_dict = default_loss_weights,
        ):
        super().__init__()
        self.transformer_config = transformer_config
        self.pooling_method = pooling_method
        self.masktoken_stats = masktoken_stats
        self.r = r
        self.loss_weight_dict = loss_weight_dict

        self.encoder_main = Transformer(**transformer_config)
        self.encoder_pret = Transformer(**transformer_config)
        self.encoder_t = Transformer(**transformer_config)
    
        self.d_z = transformer_config['dim_val']
        self.mask_generator = MaskGeneratorForecasting(d_z = self.d_z) 
            #max_len = max_len)
        self.gsat_loss_fn = GSATLoss(r = r)
        self.connected_loss = ConnectLoss()

        self.set_config()

    def forward(self, src, times = None, captum_input = False):
        '''
        Need API to match that of the trainer
        Expect: 
            src = (B, T, d = 1)
            times = None (not provided)
        '''
        out, z_seq_main, src_input = self.encoder_main(src, get_embedding = True, get_pos_input = True)
        # out shape = (B, pred_len, d = 1)
        # z_seq_main shape = (B, T, d_z)

        _, z_seq_pret = self.encoder_pret(src, get_embedding = True)

        mask_in, ste_mask = self.mask_generator(z_seq_pret, src_input)

        src_masked = self.apply_mask(src, ste_mask)

        out_masked, z_seq_masked = self.encoder_t(src_masked, get_embedding = True)

        # Need to pool z_seq's before output to be compatible with MBC
        if self.pooling_method == 'max':
            z_main = z_seq_main.max(dim=1)[0] # Across time dimension
            z_mask = z_seq_masked.max(dim=1)[0]

        total_out_dict = {
            'pred': out, # Prediction on regular embedding (prediction branch)
            'pred_mask': out_masked, # Prediction on masked embedding
            'mask_logits': mask_in, # Mask logits, i.e. before reparameterization + ste
            'ste_mask': ste_mask,
            'smooth_src': src,                                  # Keep for visualizers
            'all_z': (z_main, z_mask),
            'z_mask_list': z_mask,
        }

        return total_out_dict

    def apply_mask(self, src, ste_mask):
        #import ipdb; ipdb.set_trace()
        # First apply mask directly on input:
        ste_mask_rs = ste_mask.unsqueeze(-1)
        baseline = self._get_baseline(B = src.shape[0]).to(src.device)
        src_masked = src * ste_mask_rs + (1 - ste_mask_rs) * baseline
        return src_masked

    def _get_baseline(self, B):
        mu, std = self.masktoken_stats
        samp = torch.stack([torch.normal(mean = mu, std = std) for _ in range(B)], dim = 0).float()
        return samp

    def get_saliency_explanation(self, src, times, captum_input = False):
        pass

    def compute_loss(self, output_dict):
        mask_loss = self.loss_weight_dict['gsat'] * self.gsat_loss_fn(output_dict['mask_logits']) + self.loss_weight_dict['connect'] * self.connected_loss(output_dict['mask_logits'].unsqueeze(-1))
        return mask_loss

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)
    
    def set_config(self):
        self.config = {
            'transformer_config': self.transformer_config,
            'pooling_method': self.pooling_method,
            'r': self.r,
            'masktoken_stats': self.masktoken_stats
        }