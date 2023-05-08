import torch
import math
from torch import nn
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother, exponential_smoother
#from txai.models.mask_generators.gumbelmask_model import STENegInf
from txai.utils.functional import transform_to_attn_mask
from txai.models.encoders.positional_enc import PositionalEncodingTF

MAX = 10000.0

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

class STENegInf(torch.autograd.Function):
    # From: https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    @staticmethod
    def forward(ctx, input):
        mfill = torch.zeros_like(input).masked_fill(input < 0.5, -1e9).to(device)
        # Collapse down to sequence-level:
        mfill = mfill.sum(-1)
        return mfill

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output.unsqueeze(-1).expand(-1, -1, 4))

class CycleTrendGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            trend_smoother = True,
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
        self.trend_smoother = trend_smoother
        
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
        mask_in = (torch.sin(alpha * times.transpose(0, 1) + beta) - thresh)

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

        if self.trend_smoother:
            p = self.trend_net(agg_z).sigmoid() * self.max_len
        else:
            p = torch.zeros_like(thresh) + 1e-9

        ste_mask = STEThreshold.apply(mask_in)
        # print('src', src.shape)
        #print('p', p.shape)
        # Transpose both src and times below bc expecting batch-first input

        if self.trend_smoother:
            smooth_src = smoother(src, times, p, mask = ste_mask)
        else:
            smooth_src = src

        return smooth_src, mask_in, (alpha, beta, thresh, p)

class MaskGenStochastic(nn.Module):
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
        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        # self.alpha_net = nn.Sequential(
        #     nn.Linear(self.pre_agg_mlp_d_z, self.alpha_net_d_z),
        #     nn.PReLU(),
        #     nn.Linear(self.alpha_net_d_z, 1),
        # )

        # self.beta_net = nn.Sequential(
        #     nn.Linear(self.pre_agg_mlp_d_z, self.beta_net_d_z),
        #     nn.PReLU(),
        #     nn.Linear(self.beta_net_d_z, 1),
        # )

        # self.thresh_net = nn.Sequential(
        #     nn.Linear(self.pre_agg_mlp_d_z, self.thresh_net_d_z),
        #     nn.PReLU(),
        #     nn.Linear(self.thresh_net_d_z, 1),
        # )

        self.trend_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.trend_net_d_z),
            nn.PReLU(),
            nn.Linear(self.trend_net_d_z, 1),
        )

        self.cycle_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, 3),
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

    def mask_in_sine_curve(self, times, alpha, beta, thresh):
        mask_in = (torch.sin(alpha * times.transpose(0, 1) + beta) - thresh)
        return mask_in * 0.25 + 0.5

    def reparameterize(self, total_mask):

        if self.training:

            inv_probs = 1 - total_mask
            total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)

            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob), hard = True)[...,1]
        
        else: # No need for stochasticity, just deterministic
            total_mask_reparameterize = (total_mask > 0.5).float().squeeze(-1)

        #print('total_mask_reparameterize', total_mask_reparameterize.shape)

        return total_mask_reparameterize

    def forward(self, z_seq, src, times, get_tilde_mask = False):

        z_pre_agg = self.pre_agg_net(z_seq)

        p_time = self.time_prob_net(z_seq).sigmoid() # Sigmoid for probability
        #print('ptime', p_time.shape)
        
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        #alpha = self.alpha_net(agg_z).relu() + (2 * math.pi / self.max_len) # Should change parameterization later
        #beta = self.beta_net(agg_z).tanh() * math.pi # Map to [-pi, pi]
        #thresh = self.thresh_net(agg_z).tanh() # Map to [-1,1]

        cycle_out = self.cycle_net(agg_z)
        alpha = cycle_out[...,0].relu().unsqueeze(-1) + (2 * math.pi / self.max_len)
        beta = cycle_out[...,1].tanh().unsqueeze(-1) * math.pi
        thresh = cycle_out[...,2].tanh().unsqueeze(-1)

        p_cycle = self.mask_in_sine_curve(times, alpha, beta, thresh).unsqueeze(-1) # Masks-in values
        #print('p_cycle', p_cycle.shape)

        if self.trend_smoother:
            p = self.trend_net(agg_z).sigmoid() * self.max_len
        else:
            p = torch.zeros_like(thresh) + 1e-9

        # total_mask = (p_cycle * p_time.transpose(0,1)) # Multiplication approximates AND
        # print('tm', total_mask.max(), total_mask.min())

        # total_mask_reparameterize = self.reparameterize(total_mask)

        pc_re = self.reparameterize(p_cycle)
        pt_re = self.reparameterize(p_time.transpose(0,1))

        total_mask = p_time.transpose(0,1)

        total_mask_reparameterize = pc_re * pt_re 

        if get_tilde_mask:
            # Opposite of mask (pre-reparameterization):
            pc_re_tilde = self.reparameterize(1 - p_cycle)
            pt_re_tilde = self.reparameterize(1 - p_time.transpose(0,1))
            total_mask_tilde = pc_re_tilde * pt_re_tilde

        # Transpose both src and times below bc expecting batch-first input

        if self.trend_smoother:
            smooth_src = smoother(src, times, p, mask = total_mask_reparameterize)
        else:
            smooth_src = src

        # TODO: Get time and cycle returns later

        if get_tilde_mask:
            return smooth_src, total_mask,  total_mask_reparameterize, total_mask_tilde, (alpha, beta, thresh, p)
        else:
            return smooth_src, total_mask, total_mask_reparameterize, (alpha, beta, thresh, p)

trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}

class MaskGenStochasticDecoder(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = True,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            alpha_net_d_z = 32,
            beta_net_d_z = 32,
            thresh_net_d_z = 64,
            trend_net_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
        ):
        super(MaskGenStochasticDecoder, self).__init__()

        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.alpha_net_d_z = alpha_net_d_z
        self.beta_net_d_z = beta_net_d_z
        self.thresh_net_d_z = thresh_net_d_z
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

        self.trend_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.trend_net_d_z),
            nn.PReLU(),
            nn.Linear(self.trend_net_d_z, 1),
        )

        self.cycle_net = nn.Sequential(
            nn.Linear(self.pre_agg_mlp_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, self.thresh_net_d_z),
            nn.PReLU(),
            nn.Linear(self.thresh_net_d_z, 3),
        )

        self.time_prob_net = nn.Sequential(
            nn.Linear(d_z, 2),
        )

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

    def mask_in_sine_curve(self, times, alpha, beta, thresh):
        mask_in = (torch.sin(alpha * times.transpose(0, 1) + beta) - thresh)
        return mask_in * 0.25 + 0.5

    def reparameterize(self, total_mask):

        #if self.training:

        if total_mask.shape[-1] == 1:
            # Need to add extra dim:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)

        else:
            total_mask_prob = total_mask.softmax(dim=-1)
            #print('total_mask_prob', total_mask_prob.shape)

        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = True)[...,1]
        
        # else: # No need for stochasticity, just deterministic
        #     if total_mask.shape[-1] == 1:
        #         total_mask_reparameterize = (total_mask > 0.5).float().squeeze(-1)
        #     else:
        #         total_mask_reparameterize = total_mask.softmax(dim=-1).argmax(dim=-1).float()
            #total_mask_reparameterize = (total_mask > 0.5).float().squeeze(-1)

        #print('total_mask_reparameterize', total_mask_reparameterize.shape)

        return total_mask_reparameterize

    def forward(self, z_seq, src, times, get_tilde_mask = False):

        x = torch.cat([src, self.pos_encoder(times)], dim = -1)

        # print('x', x.shape)
        # print('z_seq', z_seq.shape)

        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq)

        z_pre_agg = self.pre_agg_net(z_seq_dec)

        p_time = self.time_prob_net(z_seq_dec) # Sigmoid for probability
        #print('ptime', p_time.shape)
        
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        #alpha = self.alpha_net(agg_z).relu() + (2 * math.pi / self.max_len) # Should change parameterization later
        #beta = self.beta_net(agg_z).tanh() * math.pi # Map to [-pi, pi]
        #thresh = self.thresh_net(agg_z).tanh() # Map to [-1,1]

        cycle_out = self.cycle_net(agg_z)
        alpha = cycle_out[...,0].relu().unsqueeze(-1) + (2 * math.pi / self.max_len)
        beta = cycle_out[...,1].tanh().unsqueeze(-1) * math.pi
        thresh = cycle_out[...,2].tanh().unsqueeze(-1)

        p_cycle = self.mask_in_sine_curve(times, alpha, beta, thresh).unsqueeze(-1) # Masks-in values
        #print('p_cycle', p_cycle.shape)

        if self.trend_smoother:
            p = self.trend_net(agg_z).sigmoid() #* self.max_len
        else:
            p = torch.zeros_like(thresh) + 1e-9

        # total_mask = (p_cycle * p_time.transpose(0,1)) # Multiplication approximates AND
        # print('tm', total_mask.max(), total_mask.min())

        # total_mask_reparameterize = self.reparameterize(total_mask)

        pc_re = self.reparameterize(p_cycle)
        pt_re = self.reparameterize(p_time.transpose(0,1))

        total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1]

        total_mask_reparameterize = pc_re * pt_re 

        if get_tilde_mask:
            # Opposite of mask (pre-reparameterization):
            pc_re_tilde = self.reparameterize(1 - p_cycle)
            pt_re_tilde = self.reparameterize(1 - p_time.transpose(0,1))
            total_mask_tilde = pc_re_tilde * pt_re_tilde

        # Transpose both src and times below bc expecting batch-first input

        if self.trend_smoother:
            smooth_src = exponential_smoother(src, times, p)
        else:
            smooth_src = src

        # TODO: Get time and cycle returns later

        if get_tilde_mask:
            return smooth_src, total_mask,  total_mask_reparameterize, total_mask_tilde, (alpha, beta, thresh, p)
        else:
            return smooth_src, total_mask, total_mask_reparameterize, (alpha, beta, thresh, p)

class MaskGenStochasticDecoder_NoCycleParam(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = False,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            trend_net_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
            use_ste = True
        ):
        super(MaskGenStochasticDecoder_NoCycleParam, self).__init__()

        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.trend_net_d_z = trend_net_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.tau = tau
        self.use_ste = use_ste

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
            nn.Linear(d_z, 2),
        )

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)


    def reparameterize(self, total_mask):

        if total_mask.shape[-1] == 1:
            # Need to add extra dim:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)

        else:
            total_mask_prob = total_mask.softmax(dim=-1)
            #print('total_mask_prob', total_mask_prob.shape)

        if self.use_ste:
            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = True)[...,1]
        else:
            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = False)[...,1]

        return total_mask_reparameterize

    def forward(self, z_seq, src, times, get_tilde_mask = False, get_agg_z = False):

        x = torch.cat([src, self.pos_encoder(times)], dim = -1)

        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq)

        z_pre_agg = self.pre_agg_net(z_seq_dec)

        p_time = self.time_prob_net(z_seq_dec) # Sigmoid for probability
        #print('ptime', p_time.shape)
        
        if self.agg == 'max':
            agg_z = z_pre_agg.max(dim=0)[0]

        if self.trend_smoother:
            p = self.trend_net(agg_z) #* self.max_len
        else:
            p = torch.zeros(src.shape[1]) + 1e-9

        #p = self.trend_net(agg_z) #* self.max_len

        # total_mask_reparameterize = self.reparameterize(total_mask)
        total_mask_reparameterize = self.reparameterize(p_time.transpose(0,1))

        total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1]

        if get_tilde_mask:
            # Opposite of mask (pre-reparameterization):
            pc_re_tilde = self.reparameterize(1 - p_cycle)
            pt_re_tilde = self.reparameterize(1 - p_time.transpose(0,1))
            total_mask_tilde = pc_re_tilde * pt_re_tilde

        # Transpose both src and times below bc expecting batch-first input

        if self.trend_smoother:
            smooth_src = exponential_smoother(src, times, p)
        else:
            smooth_src = src

        # TODO: Get time and cycle returns later

        if get_tilde_mask:
            return smooth_src, total_mask,  total_mask_reparameterize, total_mask_tilde, p
        elif get_agg_z:
            return smooth_src, total_mask,  total_mask_reparameterize, p, agg_z
        else:
            return smooth_src, total_mask, total_mask_reparameterize, p


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
            trend_smoother = True
        ):
        super(Modelv2, self).__init__()

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

        z_seq = self.encoder.embed(src, times, aggregate = False, captum_input = False)

        smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[0](z_seq, src, times)

        #ste_mask = STENegInfMod.apply(mask_in)
        #ste_mask = STEThreshold.apply(mask_in)

        # Transform into attention mask:
        ste_mask = transform_to_attn_mask(ste_mask)
        #print('mask', ste_mask.shape)

        pred = self.encoder(smooth_src, times, attn_mask = ste_mask)

        return pred, mask_in, ste_mask, smoother_stats, smooth_src

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