import torch
from torch import nn
import torch.nn.functional as F

class UnstrucMaskGen(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            trend_smoother = True,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            alpha_net_d_z = 32,
            beta_net_d_z = 32,
            thresh_net_d_z = 64,
            trend_net_d_z = 32,
            time_net_d_z = 64,
        ):
        super(MaskGenStochastic, self).__init__()

        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.alpha_net_d_z = alpha_net_d_z
        self.beta_net_d_z = beta_net_d_z
        self.thresh_net_d_z = thresh_net_d_z
        self.trend_net_d_z = trend_net_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother

        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        self.trend_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.trend_net_d_z),
            nn.PReLU(),
            nn.Linear(self.trend_net_d_z, 1),
        )

        self.time_prob_net = nn.Sequential(
            nn.Linear(d_z, self.time_net_d_z),
            nn.PReLU(),
            nn.Linear(self.time_net_d_z, self.time_net_d_z),
            nn.PReLU(),
            nn.Linear(self.time_net_d_z, self.time_net_d_z),
            nn.PReLU(),
            nn.Linear(self.time_net_d_z, self.time_net_d_z),
            nn.PReLU(),
            nn.Linear(self.time_net_d_z, 1),
        )

    def reparameterize(self, total_mask):

        if self.training:

            inv_probs = 1 - total_mask
            total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)

            total_mask_reparameterize = F.gumbel_softmax(total_mask_prob, hard = True)[...,1]
        
        else: # No need for stochasticity, just deterministic
            total_mask_reparameterize = (total_mask > 0.5).float().squeeze(-1)

        #print('total_mask_reparameterize', total_mask_reparameterize.shape)

        return total_mask_reparameterize

    def forward(self, z_seq, src, times, get_tilde_mask = False):

        x = torch.cat([src, self.pos_encoder(times)], dim = -1)

        # print('x', x.shape)
        # print('z_seq', z_seq.shape)

        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq)

        z_pre_agg = self.pre_agg_net(z_seq_dec)

        p_time = self.time_prob_net(z_seq).sigmoid() # Sigmoid for probability
        #print('ptime', p_time.shape)
        
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        if self.trend_smoother:
            p = self.trend_net(agg_z).sigmoid() * self.max_len
        else:
            p = torch.zeros_like(thresh) + 1e-9

        total_mask_reparameterize = self.reparameterize(p_time.transpose(0,1))

        total_mask = p_time.transpose(0,1)

        if get_tilde_mask:
            # Opposite of mask (pre-reparameterization):
            pt_re_tilde = self.reparameterize(1 - p_time.transpose(0,1))
            total_mask_tilde = pt_re_tilde

        # Transpose both src and times below bc expecting batch-first input

        if self.trend_smoother:
            smooth_src = smoother(src, times, p, mask = total_mask_reparameterize)
        else:
            smooth_src = src

        # TODO: Get time and cycle returns later

        if get_tilde_mask:
            return smooth_src, total_mask,  total_mask_reparameterize, total_mask_tilde
        else:
            return smooth_src, total_mask, total_mask_reparameterize