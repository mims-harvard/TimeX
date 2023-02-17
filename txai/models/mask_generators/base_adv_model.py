import torch
from torch import nn

from txai.models.base_mask_model import MaskModel
from txai.models.gumbelmask_model import STENegInf
from txai.models.gumbelmask_model import GumbelGate

from ..encoders.positional_enc import PositionalEncodingTF

from ..layers import TransformerEncoderInterpret, TransformerEncoderLayerInterpret
from ..extractors.dualattn import DualAttentionTransformer

def aggreg_transformer_output(Z, lengths, joint_mask, aggreg = 'max'):
    # Transformer embeddings through MLP --------------------------------------
    mask2 = joint_mask.permute(1, 0).unsqueeze(2).long()

    # Aggregation scheme:
    if aggreg == 'mean':
        lengths2 = lengths.unsqueeze(1)
        Z = torch.sum(Z * (1 - mask2), dim=0) / (lengths2 + 1)
    elif aggreg == 'max':
        Z, _ = torch.max(Z * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

    return Z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Outline:
1. g_phi - extractor  - GumbelExtractor
2. f_theta - predictor (Transformer simple)
3. f_theta^tilde - compliment predictor (Transformer simple)
'''

class GumbelExtractor(MaskModel):
    '''
    Built for adversarial model, will work into the codebase later
    '''

    def __init__(self, **kwargs):

        super(GumbelExtractor, self).__init__(**kwargs)

        self.MaskGate = GumbelGate(in_features = (self.d_fi + self.d_inp), out_features = self.d_inp)

        self.sensor_net = DualAttentionTransformer(
            T = self.max_len,
            d = self.d_inp,
            d_pe = self.d_pe,
            nhead = self.nhead,
            nlayers = self.nlayers,
            dim_feedforward = self.trans_dim_feedforward,
            dropout = self.trans_dropout,
            batch_first = False,
            cnn_kernel_len = 5,
            cnn_channels = 32
        )

    def enc_phi(self, src, times, static = None, captum_input = False):
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1)#.transpose(1, 2) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            if captum_input:
                src = src.unsqueeze(dim=0)
            else:
                src = src.unsqueeze(dim=1)

        lengths = torch.sum(times > 0, dim=0)
        maxlen, batch_size = src.shape[0], src.shape[1]

        pe = self.pos_encoder(times).to(device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if static is not None:
            emb = self.emb(static)

        # mask out the all-zero rows
        overlen_mask = ( torch.arange(maxlen).unsqueeze(0) >= (lengths.cpu().unsqueeze(-1)) ).to(device)

        if overlen_mask.dim() == 1:
            # Unsqueeze if only using one example (must have B first in dim.)
            overlen_mask = overlen_mask.unsqueeze(dim=1)

        # Transformer must have (T, B, d)
        # Mask is (B, T)
        # print('x', x.shape)
        # print('src', src.shape)
        output = self.sensor_net(x, src, src_key_padding_mask = overlen_mask)

        return output, overlen_mask

    def generate_mask(self, enc_phi_out, src = None):
        '''
        Params:
            enc_phi_out (tensor): outputs of enc_phi encoder
        '''
        
        # Input to MaskGate must be (B,T,d)
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

    def apply_masks(self, src, times, mask, captum_input = False):
        masked_src, masked_times = self.apply_mask(src, times, mask = mask, captum_input = captum_input)
        masked_src_tilde, masked_times_tilde = self.apply_mask(src, times, mask = (1 - mask), captum_input = captum_input)
        return masked_src, masked_times, masked_src_tilde, masked_times_tilde

    def forward(self, 
        src, 
        times,
        static = None,
        mask = None,
        captum_input = False,
        adversarial_iteration = False):

        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        # Centralized extractor computation
        out_phi, overlen_mask = self.enc_phi(src, times, static = static, captum_input = False)

        #print('overlen', overlen_mask.shape)

        if mask is None:
            if self.type_archmask == 'attention' or self.type_archmask == 'attn':
                mask, logits, attn_mask = self.generate_mask(out_phi, src)
            else:
                mask, logits = self.generate_mask(out_phi, src) # Will use sensor net in here if needed
                attn_mask = None

        inv_overlen = (~(overlen_mask)).unsqueeze(-1).repeat(1,1,mask.shape[-1]).float()
        joint_mask = mask * inv_overlen

        # Masks applied to inputs:
        #masked_src, masked_times = self.apply_mask(src, times, mask = joint_mask, captum_input = False)
        #masked_src_tilde, masked_times_tilde = self.apply_mask(src, times, mask = (1 - joint_mask), captum_input = False)

        return mask, logits, joint_mask

def compose_adv_model(extractor, predictor):
    def forward(X, times, captum_input = False):
        mask, _, joint_mask = extractor(X, times, captum_input = captum_input)
        masked_src, masked_times = extractor.apply_mask(X, times, mask = joint_mask, captum_input = (captum_input))
        output = predictor(masked_src, masked_times, captum_input = True)
        return output, mask

    return forward

class AdvMaskModel(MaskModel):

    def __init__(self, **kwargs):

        super(AdvMaskModel, self).__init__(**kwargs)

        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + self.d_inp, 
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        self.predictor = TransformerEncoderInterpret(encoder_layers, self.nlayers)
        self.mlp_predictor = nn.Sequential(
            nn.Linear(self.d_fi, self.d_fi),
            nn.ReLU(),
            nn.Linear(self.d_fi, self.n_classes),
        )

        # Identical architecture for adversarial agent:
        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + self.d_inp, 
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        self.predictor_tilde = TransformerEncoderInterpret(encoder_layers, self.nlayers)
        self.mlp_predictor_tilde = nn.Sequential(
            nn.Linear(self.d_fi, self.d_fi),
            nn.ReLU(),
            nn.Linear(self.d_fi, self.n_classes),
        )

        self.MaskGate = GumbelGate(in_features = (self.d_fi + self.d_inp), out_features = self.d_inp)

        self.sensor_net = DualAttentionTransformer(
            T = self.max_len,
            d = self.d_inp,
            d_pe = self.d_pe,
            nhead = self.nhead,
            nlayers = self.nlayers,
            dim_feedforward = self.trans_dim_feedforward,
            dropout = self.trans_dropout,
            batch_first = False,
            cnn_kernel_len = 5,
            cnn_channels = 32
        )

    def freeze_predictor(self):
        '''
        Freezes predictor module
        '''
        # Freeze enc_phi, predictor, predictor mlp
        for param in self.transformer_encoder1.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = False

        for param in self.mlp_predictor.parameters():
            param.requires_grad = False

        # Unfreeze adversarial model:
        for param in self.predictor_tilde.parameters():
            param.requires_grad = True

        for param in self.mlp_predictor_tilde.parameters():
            param.requires_grad = True

    def freeze_predictor_tilde(self):
        '''
        Freezes adversarial module
        '''
        for param in self.predictor_tilde.parameters():
            param.requires_grad = False

        for param in self.mlp_predictor_tilde.parameters():
            param.requires_grad = False

        # Unfreeze predictor:
        for param in self.predictor.parameters():
            param.requires_grad = True

        for param in self.mlp_predictor.parameters():
            param.requires_grad = True

        # Unfreeze enc_phi
        for param in self.transformer_encoder1.parameters():
            param.requires_grad = True

    def reset_freezes(self):
        for param in self.predictor_tilde.parameters():
            param.requires_grad = True

        for param in self.mlp_predictor_tilde.parameters():
            param.requires_grad = True

        # Unfreeze predictor:
        for param in self.predictor.parameters():
            param.requires_grad = True

        for param in self.mlp_predictor.parameters():
            param.requires_grad = True

    def enc_phi(self, src, times, static = None, captum_input = False):
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1)#.transpose(1, 2) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            if captum_input:
                src = src.unsqueeze(dim=0)
            else:
                src = src.unsqueeze(dim=1)

        lengths = torch.sum(times > 0, dim=0)
        maxlen, batch_size = src.shape[0], src.shape[1]

        pe = self.pos_encoder(times).to(device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if static is not None:
            emb = self.emb(static)

        # mask out the all-zero rows
        overlen_mask = ( torch.arange(maxlen).unsqueeze(0) >= (lengths.cpu().unsqueeze(-1)) ).to(device)

        if overlen_mask.dim() == 1:
            # Unsqueeze if only using one example (must have B first in dim.)
            overlen_mask = overlen_mask.unsqueeze(dim=1)

        # Transformer must have (T, B, d)
        # Mask is (B, T)
        output = self.sensor_net(x, src, src_key_padding_mask = overlen_mask)

        return output, overlen_mask

    def generate_mask(self, enc_phi_out, src = None):
        '''
        Params:
            enc_phi_out (tensor): outputs of enc_phi encoder
        '''
        
        # Input to MaskGate must be (B,T,d)
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

    def enc_theta(self, 
            src, 
            times,
            src_tilde,
            times_tilde,
            overlen_mask = None, 
            attn_mask = None, 
            captum_input = False, 
            static = None,
            adversarial_iteration = False):
        '''
        Need to ensure in overlen_mask that 1 masks out, 0 masks in
        '''
        
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 
            if overlen_mask is not None:
                overlen_mask = overlen_mask.transpose(0,1)

            times_tilde = times_tilde.transpose(0,1)
            src_tilde = src_tilde.transpose(0,1)

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if overlen_mask is None:
            # mask out the all-zero rows
            #overlen_mask = ( torch.arange(maxlen).unsqueeze(0) >= (lengths.cpu().unsqueeze(-1)) ).cuda()
            overlen_mask = (torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])).to(device)

            if overlen_mask.dim() == 1:
                # Unsqueeze if only using one example (must have B first in dim.)
                overlen_mask = overlen_mask.unsqueeze(dim=1)

        pe = self.pos_encoder(times).to(device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        pe_tilde = self.pos_encoder(times_tilde).to(device)
        x_tilde = torch.cat([pe_tilde, src_tilde], dim = 2)

        if static is not None:
            emb = self.emb(static)

        # Transformer must have (T, B, d)
        # Mask is (B, T)

        # print('x', x_tilde.shape)
        # print('overlen_mask', overlen_mask.shape)
    
        if self.type_archmask == 'attention' or self.type_archmask == 'attn':
            Z_tilde, org_attn = self.predictor_tilde(x_tilde, mask = attn_mask)
        else:
            Z_tilde, org_attn = self.predictor_tilde(x_tilde, src_key_padding_mask = overlen_mask)

        if not adversarial_iteration: # Evaluate predictor only when in regular pass
            if self.type_archmask == 'attention' or self.type_archmask == 'attn':
                Z_pred, org_attn = self.predictor(x, mask = attn_mask)
            else:
                Z_pred, org_attn = self.predictor(x, src_key_padding_mask = overlen_mask)

        Z_tilde = aggreg_transformer_output(Z_tilde, lengths, overlen_mask, aggreg = self.aggreg)

        # Feed through MLP:
        if static is not None: # Use embedding of static vector:
            Z_tilde = torch.cat([Z_tilde, emb], dim=1)

        output_tilde = self.mlp_predictor_tilde(Z_tilde)

        if not adversarial_iteration:

            Z = aggreg_transformer_output(Z_pred, lengths, overlen_mask, aggreg = self.aggreg)

            # Feed through MLP:
            if static is not None: # Use embedding of static vector:
                Z = torch.cat([Z, emb], dim=1)

            output = self.mlp_predictor(Z)

        else:
            output = None

        return output, output_tilde

    def forward(self, 
        src, 
        times,
        static = None,
        mask = None,
        captum_input = False,
        adversarial_iteration = False):

        if captum_input:
            src = src.transpose(0,1)
            times = times.transpose(0,1)

        if self.training: # Don't modify if not training
            if adversarial_iteration:
                self.freeze_predictor()
            else:
                self.freeze_predictor_tilde()

        # Centralized extractor computation
        out_phi, overlen_mask = self.enc_phi(src, times, static = static, captum_input = False)

        #print('overlen', overlen_mask.shape)

        if mask is None:
            if self.type_archmask == 'attention' or self.type_archmask == 'attn':
                mask, logits, attn_mask = self.generate_mask(out_phi, src)
            else:
                mask, logits = self.generate_mask(out_phi, src) # Will use sensor net in here if needed
                attn_mask = None

        inv_overlen = (~(overlen_mask)).unsqueeze(-1).repeat(1,1,mask.shape[-1]).float()
        joint_mask = mask * inv_overlen

        # Masks applied to inputs:
        masked_src, masked_times = self.apply_mask(src, times, mask = joint_mask, captum_input = False)
        masked_src_tilde, masked_times_tilde = self.apply_mask(src, times, mask = (1 - joint_mask), captum_input = False)

        # print('masked_src', masked_src.shape)
        # print('masked_times', masked_times.shape)
        # exit()

        output, output_tilde = self.enc_theta(
                masked_src, 
                masked_times, 
                masked_src_tilde,
                masked_times_tilde,
                overlen_mask.transpose(0,1), 
                captum_input = True, 
                attn_mask = attn_mask,
                adversarial_iteration = adversarial_iteration)

        if self.training:
            return output, output_tilde, mask, logits
        else:
            return output, mask
