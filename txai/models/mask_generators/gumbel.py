import torch
from torch import nn

from txai.models.mask_generators.gumbelmask_model import GumbelGate, STENegInf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GumbelMask(nn.Module):
    '''
    This is a simplified version of GumbelMask and MaskModel for use in the CBM
    TODO: Add support for multiple masks directly in this module
    '''
    def __init__(self,
            d_z_inp,
            d_src_inp,
            max_len = None,
            type_masktoken = 'dyna_norm_datawide',
            type_archmask = None,
            masktoken_kwargs = {},
            seed = None,
            smooth_concepts = False,
        ):

        super(GumbelMask, self).__init__()

        self.d_z_inp = d_z_inp
        self.d_src_inp = d_src_inp
        self.type_archmask = type_archmask
        self.max_len = max_len

        # Make Gate module:
        self.MaskGate = GumbelGate(in_features = self.d_z_inp, out_features = self.d_src_inp)

        # All below code migrated from previous work, we won't use most of these baselines:
        self.type_masktoken = type_masktoken
        self.masktoken_kwargs = masktoken_kwargs
        self.seed = seed
        if self.type_masktoken == 'zero':
            self.mask_token = torch.zeros(self.max_len, d_src_inp).to(device)
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

        self.smooth_concepts = smooth_concepts
        if self.smooth_concepts:
            self.moving_avg_layer = torch.nn.AvgPool1d( # Don't include parameter if we're not smoothing concepts
                kernel_size = 5,
                padding = 5 // 2,
                stride = 1,
                count_include_pad = False,
                )

    
    def generate_mask(self, z, src = None):
        '''
        Params:
            z (tensor): outputs of enc_phi encoder
        '''
        
        # Input to MaskGate must be (B,T,d)
        # Transpose to fit (B,T,d) shape
            
        mask, logits = self.MaskGate(z, training = self.training)

        # Generate mask based on attention:
        if self.type_archmask == 'attention' or self.type_archmask == 'attn':
            attn_mask = STENegInf.apply(logits).transpose(0,1) # All are negative inf coming out of here
            # Expand to SxS size:
            #print('attn_mask', attn_mask.shape)
            attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, attn_mask.shape[1])
            attn_mask = torch.add(attn_mask, attn_mask.transpose(1, 2)) # Masked-out parts should stretch across matrix by rows and columns
            return mask, logits, attn_mask

        return mask, logits

    def apply_mask(self, src, times, mask, captum_input = False):
        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        # Leaves src, times as B,T,d
        to_replace = self.get_to_replace(src, times)

        if len(mask.shape) < 2:
            M = mask.unsqueeze(-1).repeat(1,1,src.shape[-1])
        else:
            M = mask

        # print('Src', src.shape)
        # print('M', M.shape)
        # print('to_replace', to_replace.shape)

        if self.smooth_concepts:
            src = self.moving_avg_layer(src) # Smooth before replacement
        
        src = src * M + to_replace * (1 - M)

        # TODO: gate times (need for cyclic and time-specificity concepts)

        return src, times

    def get_to_replace(self, src, times):
        if self.type_masktoken == 'dynamic_normal':
            # Generate new mask:
            # Take sensor-wise mu, std across the sample:
            mu = torch.mean(src, dim=1).unsqueeze(1).repeat(1, src.shape[1], 1)
            std = torch.std(src, dim=1, unbiased = True).unsqueeze(1).repeat(1, src.shape[1], 1)
            to_replace = mu + std * torch.rand_like(src)
        elif self.type_masktoken == 'dyna_norm_datawide':
            to_replace = self.mask_token().unsqueeze(1).repeat(1, src.shape[1], 1)
            #to_replace = torch.stack([self.mask_token() for _ in range(src.shape[1])], dim = 1)
        elif self.type_masktoken == 'norm_datawide' or self.type_masktoken == 'zero':
            to_replace = self.mask_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        elif (self.type_masktoken == 'decomp_dyna') or (self.type_masktoken == 'decomp_zero'):
            trend, sea = self.mask_token()
            trend = trend.unsqueeze(0).repeat(src.shape[0], 1, 1)
            sea = sea.unsqueeze(0).repeat(src.shape[0], 1, 1)
            to_replace = (trend, sea)
        else:
            to_replace = self.mask_token.unsqueeze(-1).unsqueeze(0).repeat(src.shape[0], 1, src.shape[-1])

        return to_replace


    def forward(self,
            src,
            times,
            z,
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

        if self.type_archmask == 'attention' or self.type_archmask == 'attn':
            mask, logits, attn_mask = self.generate_mask(z, src.transpose(0, 1) if captum_input else src)
        else:
            mask, logits = self.generate_mask(z, src.transpose(0, 1) if captum_input else src) 
            # Will use sensor net in here if needed
            attn_mask = None

        # Actually multiplies the mask out with the baseline
        masked_src, masked_times = self.apply_mask(src, times, mask = mask, captum_input = captum_input)

        return masked_src, masked_times, mask, logits, attn_mask