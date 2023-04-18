import torch
import math
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother, exponential_smoother
from txai.models.mask_generators.gumbelmask_model import STENegInf
from txai.utils.functional import transform_to_attn_mask
from txai.models.encoders.positional_enc import PositionalEncodingTF

trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}
MAX = 10000.0

class MaskGenWindow(nn.Module):
    '''
    Currently only works for univariate
    '''
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = True,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            trend_net_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
        ):
        super(MaskGenWindow, self).__init__()

        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.trend_net_d_z = trend_net_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.tau = tau

        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        # Predicts smoothing parameter:
        self.trend_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.trend_net_d_z),
            nn.PReLU(),
            nn.Linear(self.trend_net_d_z, 1),
        )

        self.L_time_prob_net = nn.Sequential(
            nn.Linear(d_z, 1),
        )

        self.R_time_prob_net = nn.Sequential(
            nn.Linear(d_z, 1),
        )

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

    def reparameterize(self, T_prob):

        if self.training:
            one_side_reparameterize = F.gumbel_softmax(torch.log(T_prob + 1e-9), tau = self.tau, hard = True)
        else:
            # Perform discrete argmax (no reparameterization)
            amax = T_prob.argmax(dim=-1)
            one_side_reparameterize = torch.zeros_like(T_prob)
            one_side_reparameterize[:,amax] = 1

        return one_side_reparameterize

    def make_full_mask(self, phi_hat_L, phi_hat_R):

        B, T = phi_hat_L.shape
        assert B == phi_hat_R.shape[0]

        #lT = torch.tril(torch.ones(T, T)).unsqueeze(0).repeat(B,1,1)
        L_n = torch.triu(torch.ones(T, T)).to(phi_hat_L.device)

        # Apply equation 9 from Dynamic Window Attention paper:
        left_side = torch.matmul(phi_hat_L, L_n) * torch.matmul(phi_hat_R, L_n.transpose(0, 1))
        right_side = torch.matmul(phi_hat_R, L_n) * torch.matmul(phi_hat_L, L_n.transpose(0, 1))

        return (left_side + right_side)


    def forward(self, z_seq, src, times, get_tilde_mask = False):

        x = torch.cat([src, self.pos_encoder(times)], dim = -1)

        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq)

        z_pre_agg = self.pre_agg_net(z_seq_dec)

        phi_L = self.L_time_prob_net(z_seq_dec).squeeze().transpose(0,1).softmax(dim=-1) # (B, T) for both
        phi_R = self.R_time_prob_net(z_seq_dec).squeeze().transpose(0,1).softmax(dim=-1)

        # print('phi L', phi_L.shape)
        # print('phi R', phi_R.shape)
        
        # Predicts smoothing parameter for learnable smoothing:
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        if self.trend_smoother:
            p = self.trend_net(agg_z).sigmoid() #* self.max_len
        else:
            p = torch.zeros(src.shape[1]) + 1e-9

        phi_hat_L = self.reparameterize(phi_L)
        phi_hat_R = self.reparameterize(phi_R)

        hard_mask = self.make_full_mask(phi_hat_L, phi_hat_R)
        smooth_mask = self.make_full_mask(phi_L, phi_R) # Used purely for optimization

        # Both above are size (B, T) - need to expand to multidimensional

        if self.trend_smoother:
            smooth_src = exponential_smoother(src, times, p)
        else:
            smooth_src = src

        # TODO: Get time and cycle returns later

        return smooth_src, smooth_mask, hard_mask, p #(phi_L, phi_R)