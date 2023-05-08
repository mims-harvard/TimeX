# Decoupled training version of SAT model

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np

import os
import sys; sys.path.append(os.path.dirname(__file__))
from ..encoders.positional_enc import PositionalEncodingTF

from ..layers import TransformerEncoderInterpret, TransformerEncoderLayerInterpret
from .base_mask_model import MaskModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_extractor(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-1e-10, 1e-10)



class GumbelGate(nn.Module):
    def __init__(self, in_features, hidden_dims = 32, out_features = 1, dropout = 0.2):
        super(GumbelGate, self).__init__()
        self.in_features = in_features
        self.hd = hidden_dims
        self.out_features = out_features

        self.gate = nn.Sequential(
            nn.Linear(self.in_features, self.hd),
            nn.ELU(),
            nn.Linear(self.hd, out_features),
        )

        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, training = True, hard_sampling = True):

        probs = self.gate(self.dropout(x)).sigmoid()
        #print('Probs shape', probs.shape)
        # Should be of size (B,T,1)

        if training:
            # Augment for gumbel-softmax resampling:
            inv_probs = 1 - probs

            if self.out_features == 1:
                # Inv_probs comes in 0th row because this treated as spot for masking-out
                # By putting probs in row 1, score computed by mask comes directly from
                #   likelihood of masking-in the given index of the input
                logits = torch.cat([inv_probs, probs], dim=-1)
            else:
                probs, inv_probs = probs.unsqueeze(-1), inv_probs.unsqueeze(-1)
                logits = torch.cat([inv_probs, probs], dim = -1)
                
            # Hard ensures that the output is discrete - something not guaranteed in KumaMask
            sampled = F.gumbel_softmax(torch.log(logits + 1e-9), hard = hard_sampling, tau = 1)
            mask = sampled[...,1] # Get only 1-activated spots
            # Still differentiable since logits was constructed with probs computation

        else:
            mask = (probs > 0.5).float() # Hard sigmoid activation
            # Non-differentiable, so only use during forward pass on testing

        # if np.random.rand() < 0.1:
        #     print('Mask size', mask.shape)

        if len(mask.shape) < 3: # Update mask to fit to univariate data
            mask = mask.unsqueeze(-1)

        return mask, probs


class GumbelMask(MaskModel):
    def __init__(self, importance_in_evals = False, get_embeddings = False, 
            predictor = None, **kwargs):

        super(GumbelMask, self).__init__(**kwargs)

        # Make Gate module:
        # if self.sensor_net:
        #     self.MaskGate = GumbelGate(in_features = (self.enc_phi.outshape), out_features = self.d_inp)
        # else:
        self.MaskGate = GumbelGate(in_features = (self.enc_phi.d_out), out_features = self.d_inp)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_fi, self.d_fi),
            nn.ReLU(),
            nn.Linear(self.d_fi, self.n_classes),
        )

        #self.mask_token = torch.zeros(self.max_len).to(device)
        self.importance_in_evals = importance_in_evals
    
    def generate_mask(self, enc_phi_out, src = None):
        '''
        Params:
            enc_phi_out (tensor): outputs of enc_phi encoder
        '''
        
        # Input to MaskGate must be (B,T,d)
        # Transpose to fit (B,T,d) shape
            
        mask, logits = self.MaskGate(enc_phi_out, training = self.training)

        # Generate mask based on attention:
        if self.type_archmask == 'attention' or self.type_archmask == 'attn':
            attn_mask = STENegInf.apply(logits.squeeze(-1)) # All are negative inf coming out of here
            # Expand to SxS size:
            attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, attn_mask.shape[1])
            #print('attn_mask.shape', attn_mask.shape)
            attn_mask = torch.add(attn_mask, attn_mask.transpose(1, 2)) # Masked-out parts should stretch across matrix by rows and columns
            #print('attn_mask.shape', attn_mask.shape)

            return mask, logits, attn_mask

        return mask, logits

    def enc_theta(self, src, times, joint_mask = None, attn_mask = None, captum_input = False, static = None, get_embeddings = False):
        '''
        Need to ensure in joint_mask that 1 masks out, 0 masks in
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

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        # print('lengths shape', lengths.shape)
        # print('times', times.shape)
        # print('src shape', src.shape)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if joint_mask is None:
            # mask out the all-zero rows
            #overlen_mask = ( torch.arange(maxlen).unsqueeze(0) >= (lengths.cpu().unsqueeze(-1)) ).cuda()
            overlen_mask = (torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])).to(device)

            if overlen_mask.dim() == 1:
                # Unsqueeze if only using one example (must have B first in dim.)
                overlen_mask = overlen_mask.unsqueeze(dim=1)
                
            joint_mask = overlen_mask

        #print('joint_mask', joint_mask.shape)

        # Encode input vectors
        # if self.use_mlp_encoder:
        #     src = self.MLP_encoder(src)

        #pe = self.pos_encoder(times.transpose(0,1)).transpose(0,1) # Positional encoder
        pe = self.pos_encoder(times).to(device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if static is not None:
            emb = self.emb(static)

        # Transformer must have (T, B, d)
        # Mask is (B, T)
    
        if self.type_archmask == 'attention' or self.type_archmask == 'attn':
            Z, org_attn = self.encoder2(x, mask = attn_mask)
        else:
            Z, org_attn = self.encoder2(x, src_key_padding_mask = joint_mask)
        #print('output shape', output.shape)

        # Transformer embeddings through MLP --------------------------------------
        mask2 = joint_mask.permute(1, 0).unsqueeze(2).long()
        #print('mask2', mask2)

        # Aggregation scheme:
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            Z = torch.sum(Z * (1 - mask2), dim=0) / (lengths2 + 1)
        elif self.aggreg == 'max':
            Z, _ = torch.max(Z * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        # Feed through MLP:
        if static is not None: # Use embedding of static vector:
            Z = torch.cat([Z, emb], dim=1)

        #print('input to mlp', output.shape)
        output = self.mlp(Z)

        #print('output', output.shape)

        if get_embeddings:
            return output, Z

        return output


    def forward(self,
            src,
            times,
            static = None,
            mask = None,
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            get_embeddings = False
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

        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        out_phi, overlen_mask = self.enc_phi(src, times, static = static, captum_input = False)

        if mask is None:
            # if self.importance_in_evals or not (self.training):
            #     mask, logits = self.generate_mask(out_phi)
            # else:
            if self.type_archmask == 'attention' or self.type_archmask == 'attn':
                mask, logits, attn_mask = self.generate_mask(out_phi, src)
            else:
                mask, logits = self.generate_mask(out_phi, src) # Will use sensor net in here if needed
                attn_mask = None

            #if not self.training:
                #print('mask', mask)
        #print('mask', mask.shape)

        inv_overlen = (~(overlen_mask)).unsqueeze(-1).repeat(1,1,mask.shape[-1]).float()
        # print('mask', mask.shape)
        # print('inv_overlen', inv_overlen.shape)
        joint_mask = mask * inv_overlen # Combine with over-length mask

        # print('joint_mask', joint_mask.shape)
        # # print('overlen_mask', overlen_mask.shape)
        # print('src', src.shape)

        masked_src, masked_times = self.apply_mask(src, times, mask = joint_mask, captum_input = False)

        # print('masked_src', masked_src.shape)
        # print('masked_times', masked_times.shape)

        if get_embeddings:
            output, Z = self.enc_theta(masked_src, masked_times, overlen_mask, captum_input = True, get_embeddings = get_embeddings, attn_mask = attn_mask)
        else:
            output = self.enc_theta(masked_src, masked_times, overlen_mask, captum_input = True, attn_mask = attn_mask)

        if self.training:
            if get_embeddings:
                return output, mask, logits, Z
            # Else, don't return embeddings:
            return output, mask, logits

        else: # We are evaluating:
            if self.importance_in_evals:
                return output, logits
            else:
                return output, mask

class GumbelMaskRD(MaskModel):
    def __init__(self, importance_in_evals = False, get_embeddings = False, raindrop_kwargs = {}, **kwargs):

        super(GumbelMaskRD, self).__init__(type_masktoken = 'zero', **kwargs)

        # Make Gate module:
        if self.sensor_net:
            self.MaskGate = GumbelGate(in_features = (self.d_fi + self.d_inp), out_features = self.d_inp)
        else:
            self.MaskGate = GumbelGate(in_features = (self.d_fi), out_features = self.d_inp)

        self.importance_in_evals = importance_in_evals

        global_structure = torch.ones(self.d_inp, self.d_inp)

        # Setup Raindrop module:
        self.encoder2 = Raindrop_v2(
            d_inp = self.d_inp,
            max_len = self.max_len,
            n_classes = self.n_classes,
            sensor_wise_mask = True,
            static = False,
            global_structure = global_structure,
            nhead = 1,
            nlayers = 2,
            d_static = 0,
            **raindrop_kwargs,
        )
    
    def generate_mask(self, enc_phi_out, src = None):
        '''
        Params:
            enc_phi_out (tensor): outputs of enc_phi encoder
        '''
        
        # Input to MaskGate must be (B,T,d)
        # Transpose to fit (B,T,d) shape
        # if self.importance_in_evals:
        #     mask, probs = self.MaskGate(enc_phi_out, training = self.training, importance = True)
        #     return mask, probs
        #else:
        if self.sensor_net:
            # Must squeeze and unsqueeze the channel
            sensor_embeds = self.snet(src.transpose(0,1).unsqueeze(dim=1)).squeeze(dim=1)
            # print('Sensor embeds', sensor_embeds.shape)
            # print('enc phi', enc_phi_out.shape)
            enc_phi_out = torch.cat([enc_phi_out, sensor_embeds], dim = -1)
        mask, logits = self.MaskGate(enc_phi_out, training = self.training)
        return mask, logits

    def enc_theta(self, src, times, joint_mask = None, captum_input = False, static = None, get_embeddings = False):
        '''
        Need to ensure in joint_mask that 1 masks out, 0 masks in

        Leaving old arguments in here for compatibility
        '''
        
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1)#.transpose(1, 2) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)

        # Raindrop must have (T, B, d)
        output, distance, _ = self.encoder2(src = src, static = static, times = times, lengths = lengths)

        return output


    def forward(self,
            src,
            times,
            static = None,
            mask = None,
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            get_embeddings = False
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

        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        out_phi, overlen_mask = self.enc_phi(src, times, static = static, captum_input = False)

        if mask is None:
            mask, logits = self.generate_mask(out_phi, src) # Will use sensor net in here if needed


        inv_overlen = (~(overlen_mask)).unsqueeze(-1).repeat(1,1,mask.shape[-1]).float()
        joint_mask = mask * inv_overlen # Combine with over-length mask


        masked_src, masked_times = self.apply_mask(src, times, mask = joint_mask, captum_input = False)

        masked_src = torch.cat([masked_src, mask], dim = 2)

        output = self.enc_theta(masked_src, masked_times, captum_input = True)

        if self.training:
            # Else, don't return embeddings:
            return output, mask, logits

        else: # We are evaluating:
            if self.importance_in_evals:
                return output, logits
            else:
                return output, mask