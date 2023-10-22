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


all_default_opt_kwargs = {
    'lr': 0.0001,
    'weight_decay': 0.01,
} 


from dataclasses import dataclass, field
@dataclass
class AblationParameters:
    equal_g_gt: bool = field(default = False)
    hard_concept_matching: bool = field(default = True)
    use_loss_on_concept_sims: bool = field(default = False)
    use_concept_corr_loss: bool = field(default = False)
    g_pret_equals_g: bool = field(default = False)
    label_based_on_mask: bool = field(default = False)
    use_ste: bool = field(default = True)
    # Prototypes:
    ptype_assimilation: bool = field(default = False)
    side_assimilation: bool = field(default = False)
    archtype: str = field(default = 'transformer')
    cnn_dim: int = field(default = 128)

default_abl = AblationParameters() # Class based on only default params

default_loss_weights = {
    'gsat': 1.0,
    'connect': 1.0,
}

class TimeXModel(nn.Module):
    '''
    Model has full options through config
        - Use for ablations - configuration supported through config load
    '''
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_prototypes,
            gsat_r, 
            transformer_args = transformer_default_args,
            ablation_parameters = default_abl,
            loss_weight_dict = default_loss_weights,
            tau = 1.0,
            masktoken_stats = None,
        ):
        super(TimeXModel, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.transformer_args = transformer_args
        self.n_prototypes = n_prototypes
        self.gsat_r = gsat_r
        self.tau = tau
        self.masktoken_stats = masktoken_stats

        self.ablation_parameters = ablation_parameters
        self.loss_weight_dict = loss_weight_dict
        
        # Holds main encoder:
        if self.ablation_parameters.archtype == 'transformer':
            self.encoder_main = TransformerMVTS(
                d_inp = d_inp,  # Dimension of input from samples (must be constant)
                max_len = max_len, # Max length of any sample to be fed into model
                n_classes = self.n_classes, # Number of classes for classification head
                **self.transformer_args
            )
            self.d_z = (self.d_inp + self.d_pe)
        elif self.ablation_parameters.archtype == 'cnn':
            self.encoder_main = CNN(
                d_inp = d_inp,
                n_classes = self.n_classes,
                dim = self.ablation_parameters.cnn_dim # Abuse of notation, but just for experiment purposes
            )
            self.d_z = self.ablation_parameters.cnn_dim
        elif self.ablation_parameters.archtype == 'lstm':
            self.encoder_main = LSTM(
                d_inp = d_inp,
                n_classes = self.n_classes,
            )
            self.d_z = 128

        self.encoder_pret = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args # TODO: change to a different parameter later - leave simple for now
        )

        if self.ablation_parameters.archtype == 'transformer':
            self.encoder_t = TransformerMVTS(
                d_inp = d_inp,  # Dimension of input from samples (must be constant)
                max_len = max_len, # Max length of any sample to be fed into model
                n_classes = self.n_classes, # Number of classes for classification head
                **self.transformer_args # TODO: change to a different parameter later - leave simple for now
            )
        elif self.ablation_parameters.archtype == 'cnn':
            self.encoder_t = CNN(
                d_inp = d_inp,
                n_classes = self.n_classes,
                dim = self.ablation_parameters.cnn_dim
            )
        elif self.ablation_parameters.archtype == 'lstm':
            self.encoder_t = LSTM(
                d_inp = d_inp,
                n_classes = self.n_classes,
            )

        # For decoder, first value [0] is actual value, [1] is mask value (predicted logit)

        self.mask_generator = MaskGenerator(d_z = (self.d_inp + self.d_pe), d_pe = self.d_pe, max_len = max_len, tau = self.tau, 
            use_ste = self.ablation_parameters.use_ste)

        # Below is a sequence-level embedding - (N_c, T, 2, d_z) -> T deals with the length of the time series
        #   - The above is done to allow for sequence-level decoding
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.d_z)) # 2 defines mu and logvar

        if self.ablation_parameters.label_based_on_mask:
            self.z_e_predictor = nn.Sequential(
                nn.Linear(self.d_z, self.d_z),
                nn.ReLU(),
                nn.Linear(self.d_z, n_classes),
            )

        # Setup loss functions:
        self.gsat_loss_fn = GSATLoss(r = self.gsat_r)
        self.connected_loss = ConnectLoss()

        self.set_config()

    def forward(self, src, times, captum_input = False):
        # TODO: return early from function when in eval
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        if self.ablation_parameters.archtype == 'transformer':
            pred_regular, z_main, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        else:
            pred_regular, z_main = self.encoder_main(src, times, captum_input = False, get_embedding = True)

        if not self.ablation_parameters.g_pret_equals_g:
            z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)
        

        # Generate smooth_src: # TODO: expand to lists
        smooth_src_list, mask_in_list, ste_mask_list, = [], [], []

        if self.ablation_parameters.g_pret_equals_g:
            mask_in, ste_mask = self.mask_generator(z_seq_main, src, times)
        else:
            mask_in, ste_mask = self.mask_generator(z_seq, src, times)

        # Need sensor-level masking if multi-variate:
        if self.d_inp > 1 or (self.ablation_parameters.archtype != 'transformer'):
            exp_src, ste_mask_attn = self.multivariate_mask(src, ste_mask)
        else:
            # Easy, simply transform to attention mask:
            ste_mask_attn = transform_to_attn_mask(ste_mask)
            exp_src = src

        if self.ablation_parameters.archtype == 'transformer':
            if self.ablation_parameters.equal_g_gt:
                pred_mask, z_mask, z_seq_mask = self.encoder_main(exp_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)
            else:
                pred_mask, z_mask, z_seq_mask = self.encoder_t(exp_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)
        else:
            if self.ablation_parameters.equal_g_gt:
                pred_mask, z_mask = self.encoder_main(exp_src, times, get_embeddding = True)
            else:
                pred_mask, z_mask = self.encoder_t(exp_src, times, get_embedding = True)

        if self.ablation_parameters.ptype_assimilation:
            ptypes, match_m = self.hard_ptype_matching(z_mask)
            if not self.training: # Only get indices if testing bc it's an expensive op
                ptype_inds = torch.cat([match_m[i,:].nonzero(as_tuple=True)[0] for i in range(match_m.shape[0])])
            else:
                ptype_inds = []
        else:
            ptype_inds = None

        if self.ablation_parameters.label_based_on_mask:
            pred_mask = self.z_e_predictor(z_mask) # Make prediction on masked input

        total_out_dict = {
            'pred': pred_regular, # Prediction on regular embedding (prediction branch)
            'pred_mask': pred_mask, # Prediction on masked embedding
            'mask_logits': mask_in, # Mask logits, i.e. before reparameterization + ste
            'ste_mask': ste_mask,
            'smooth_src': src,                                  # Keep for visualizers
            'all_z': (z_main, z_mask),
            'z_mask_list': z_mask,
        }

        if self.ablation_parameters.ptype_assimilation:
            total_out_dict['ptype_inds'] = ptype_inds
            total_out_dict['ptypes'] = ptypes

        return total_out_dict

    def get_saliency_explanation(self, src, times, captum_input = False):
        '''
        Retrieves only saliency explanation (not concepts)
            - More efficient than calling forward due to less module calls
        '''

        if self.ablation_parameters.g_pret_equals_g:
            z_seq = self.encoder_main.embed(src, times, captum_input = False, aggregate = False)
        else:
            z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        mask_in, ste_mask = self.mask_generator(z_seq, src, times)

        out_dict = {
            'smooth_src': src,
            'mask_in': mask_in.transpose(0,1),
            'ste_mask': ste_mask,
        }

        return out_dict

    def forward_pass_ge(self, src, times, ste_mask, captum_input = False):
        if self.d_inp > 1:
            exp_src, ste_mask_attn = self.multivariate_mask(src, ste_mask)
        else:
            # Easy, simply transform to attention mask:
            ste_mask_attn = transform_to_attn_mask(ste_mask)
            exp_src = src
    
        pred_mask, z_mask, z_seq_mask = self.encoder_t(exp_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)

        return pred_mask

    def multivariate_mask(self, src, ste_mask):
        # First apply mask directly on input:
        baseline = self._get_baseline(B = src.shape[1])
        ste_mask_rs = ste_mask.transpose(0,1)
        if len(ste_mask_rs.shape) == 2:
            ste_mask_rs = ste_mask_rs.unsqueeze(-1)
        # print('ste_mask_rs', ste_mask_rs.shape)
        # print('baseline', baseline.shape)
        # print('src', src.shape)
        src_masked = src * ste_mask_rs + (1 - ste_mask_rs) * baseline

        # Then deduce attention mask by multiplication across sensors:
        attention_ste_mask = transform_to_attn_mask(ste_mask)

        return src_masked, attention_ste_mask

    def hard_ptype_matching(self, z_mask):
        # z_mask: shape (B, d_z)

        if self.ablation_parameters.side_assimilation:
            zm_n = F.normalize(z_mask.detach(), dim = -1)
        else:
            zm_n = F.normalize(z_mask, dim = -1)
        pt_n = F.normalize(self.prototypes, dim = -1)

        sim_mat = torch.matmul(zm_n, pt_n.transpose(0,1))

        sim_prob_log = sim_mat.log_softmax(dim=-1) # Shape (B, N_p)

        # Hard matching:
        hard_match_matrix = F.gumbel_softmax(sim_prob_log, tau = 0.1, hard = True)

        ptype_replacements = torch.matmul(hard_match_matrix, self.prototypes)

        return ptype_replacements, hard_match_matrix

    def init_prototypes(self, train, seed = None, stratify = True):
        
        # Initialize prototypes with random training samples:
        X, times, y = train

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if stratify:
            # Create class weights for tensor:
            inds = stratified_sample(y, n = self.n_prototypes)
        else:
            inds = torch.randperm(X.shape[1])[:self.n_prototypes]

        Xp, times_p = X[:,inds,:], times[:,inds]
        if self.ablation_parameters.archtype == 'transformer':
            z_p = self.encoder_main.embed(Xp, times_p, captum_input = False)
        else:
            _, z_p = self.encoder_main(Xp, times_p, captum_input = False, get_embedding = True)

        self.prototypes = torch.nn.Parameter(z_p.detach().clone()) # Init prototypes to class (via parameter)
    
    def _get_baseline(self, B):
        mu, std = self.masktoken_stats
        samp = torch.stack([torch.normal(mean = mu, std = std) for _ in range(B)], dim = 1)
        return samp

    def compute_loss(self, output_dict):
        mask_loss = self.loss_weight_dict['gsat'] * self.gsat_loss_fn(output_dict['mask_logits']) + self.loss_weight_dict['connect'] * self.connected_loss(output_dict['mask_logits'])
        return mask_loss

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)


    def set_config(self):
        self.config = {
            'd_inp': self.d_inp,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'n_prototypes': self.n_prototypes,
            'gsat_r': self.gsat_r,
            'transformer_args': self.transformer_args,
            'ablation_parameters': self.ablation_parameters,
            'tau': self.tau,
            'masktoken_stats': self.masktoken_stats,
        }