import torch
import math
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother, exponential_smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.encoders.positional_enc import PositionalEncodingTF

trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}

MAX = 10000.0

class MaskGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = False,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
            use_ste = True
        ):
        super(MaskGenerator, self).__init__()

        self.d_z = d_z
        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.tau = tau
        self.use_ste = use_ste

        self.d_inp = self.d_z - d_pe

        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        if self.d_inp > 1:
            self.time_prob_net = nn.Sequential(nn.Linear(d_z, self.d_inp), nn.Sigmoid())
        else:
            self.time_prob_net = nn.Linear(d_z, 2)
        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        self.init_weights()

    def init_weights(self):
        def iweights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.time_prob_net.apply(iweights)
        self.pre_agg_net.apply(iweights)

    def reparameterize(self, total_mask):

        if self.d_inp == 1:
            if total_mask.shape[-1] == 1:
                # Need to add extra dim:
                inv_probs = 1 - total_mask
                total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)
            else:
                total_mask_prob = total_mask.softmax(dim=-1)
        else:
            # Need to add extra dim:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)

        #if self.training:
        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = self.use_ste)[...,1]
        # else:
        #     am = total_mask_prob.argmax(dim=-1)
        #     total_mask_reparameterize = 

        return total_mask_reparameterize

    def forward(self, z_seq, src, times, get_agg_z = False):

        if torch.any(times < -1e5):
            tgt_mask = (times < -1e5).transpose(0,1)
        else:
            tgt_mask = None

        x = torch.cat([src, self.pos_encoder(times)], dim = -1)
        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq, tgt_key_padding_mask = tgt_mask)
        z_pre_agg = self.pre_agg_net(z_seq_dec)

        p_time = self.time_prob_net(z_seq_dec)
        total_mask_reparameterize = self.reparameterize(p_time.transpose(0,1))
        if self.d_inp == 1:
            total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1].unsqueeze(-1)
        else:
            total_mask = p_time # Already sigmoid transformed

        # Transpose both src and times below bc expecting batch-first input

        # TODO: Get time and cycle returns later

        if get_agg_z:
            agg_z = z_pre_agg.max(dim=0)[0]
            return total_mask, total_mask_reparameterize, agg_z
        else:
            return total_mask, total_mask_reparameterize