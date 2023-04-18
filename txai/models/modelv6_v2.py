import torch
from torch import nn
import torch.nn.functional as F
from torch import linalg

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.modelv2 import MaskGenStochasticDecoder_NoCycleParam
from txai.models.mask_generators.window_mask import MaskGenWindow

from txai.utils.predictors.loss import GSATLoss, ConnectLoss
from txai.utils.predictors.loss_smoother_stats import *

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

# class PrototypeLayer(nn.Module):
#     def __init__(self,
#             n_prototypes,
#             d_z,    
#         ):

#         self.ptype = nn.Parameter(torch.randn(n_prototypes, d_z), requires_grad = True)
#         self.__init_weight()

#     def forward(self, z):
#         # Perform linear:
#         zout_nonorm = F.linear(z, self.ptype)
#         # Normalize so it's equivalent to cosine similarity:
#         zout = zout_nonorm / (linalg.norm(z, p=2, dim = 1) * linalg.norm(self.ptype, p=2, dim=0))
#         return zout

#     def __init_weight(self):
#         nn.init.kaiming_normal_(self.ptype)


all_default_opt_kwargs = {
    'lr': 0.0001,
    'weight_decay': 0.01,
} 

class OptimizerHelper:
    # Class to help with heterogeneous learning rates on the training process

    def __init__(self, 
            model,
            pred_branch_opt_kwargs = all_default_opt_kwargs,
            T_opt_kwargs = all_default_opt_kwargs,
            score_opt_kwargs = all_default_opt_kwargs,
            concept_opt_kwargs = all_default_opt_kwargs,
        ):

        self.pred_branch = torch.optim.AdamW(model.encoder_main.parameters(), **pred_branch_opt_kwargs)
        T_params = list(model.encoder_pret.parameters()) + list(model.mask_generators.parameters())
        self.T_optimizer = torch.optim.AdamW(T, **T_opt_kwargs)
        self.score_optimizer = torch.optim.Adamw(model.mask_score_net.parameters(), **score_opt_kwargs)
        self.concept_optimizer = torch.optim.AdamW(model.concept_dists, **concept_opt_kwargs)

    def zero_grad(self):
        self.pred_branch.zero_grad()
        self.T_optimizer.zero_grad()
        self.score_optimizer.zero_grad()
        self.concept_optimizer.zero_grad()

    def step(self):
        self.pred_branch.step()
        self.T_optimizer.step()
        self.score_optimizer.step()
        self.concept_optimizer.step()

from dataclasses import dataclass, field
@dataclass
class AblationParameters:
    equal_g_gt: bool = field(default = False)
    trend_smoother_loss: bool = field(default = True)
    concept_matching: bool = field(default = False)

default_abl = AblationParameters() # Class based on only default params

default_loss_weights = {
    'gsat': 1.0,
    'connect': 1.0,
}

class Modelv6_v2(nn.Module):
    '''
    Model has full options through config
        - Use for ablations - configuration supported through config load
    '''
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_concepts,
            n_explanations,
            gsat_r, 
            size_mask_target_val = 5,
            transformer_args = transformer_default_args,
            trend_smoother = True,
            use_window = False,
            ablation_parameters = default_abl,
            loss_weight_dict = default_loss_weights,
        ):
        super(Modelv6_v2, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.trend_smoother = trend_smoother
        self.transformer_args = transformer_args
        self.n_concepts = n_concepts
        self.n_explanations = n_explanations
        self.gsat_r = gsat_r
        self.size_mask_target_val = size_mask_target_val
        self.use_window = use_window

        self.ablation_parameters = ablation_parameters
        self.loss_weight_dict = loss_weight_dict

        d_z = (self.d_inp + self.d_pe)
        
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

        self.mask_score_net = nn.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, 1))

        # For decoder, first value [0] is actual value, [1] is mask value (predicted logit)

        self.mask_generators = nn.ModuleList()
        for _ in range(self.n_explanations):
            if use_window:
                mgen = MaskGenWindow(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother)
            else:
                mgen = MaskGenStochasticDecoder_NoCycleParam(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother)
            self.mask_generators.append(mgen)

        # Below is a sequence-level embedding - (N_c, T, 2, d_z) -> T deals with the length of the time series
        #   - The above is done to allow for sequence-level decoding
        self.concept_dists = nn.Parameter(torch.randn(self.n_concepts, d_z)) # 2 defines mu and logvar

        # Setup loss functions:
        self.gsat_loss_fn = GSATLoss(r = self.gsat_r)
        self.psizeloss = PSizeLoss()
        self.sizemaskloss = SizeMaskLoss(mean = True, target_val = self.size_mask_target_val)
        self.connected_loss = ConnectLoss()
        #self.sizemaskloss = KLDMaskLoss()

        self.set_config()

    def forward(self, src, times, captum_input = False):
        # TODO: return early from function when in eval
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_main, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        # Generate smooth_src: # TODO: expand to lists
        smooth_src_list, mask_in_list, ste_mask_list, p_list = [], [], [], []
        pred_mask_list, z_mask_list = [], []
        mask_score_list = []
        all_concept_scores, cs_inds_list, all_concepts = [], [], []
        for i in range(self.n_explanations): # Iterate over all explanations
            smooth_src, mask_in, ste_mask, p = self.mask_generators[i](z_seq, src, times, get_tilde_mask = False)
            ste_mask_attn = transform_to_attn_mask(ste_mask)

            smooth_src_list.append(smooth_src)
            mask_in_list.append(mask_in)
            ste_mask_list.append(ste_mask)
            p_list.append(p)

            if self.ablation_parameters.equal_g_gt:
                pred_mask, z_mask, z_seq_mask = self.encoder_main(smooth_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)
            else:
                pred_mask, z_mask, z_seq_mask = self.encoder_t(smooth_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)

            pred_mask_list.append(pred_mask)

            # Gives score of current explanation considered:
            score_exp_i = self.mask_score_net(z_mask) # Not trained on variational for now
            # Above should be (B,1)
            
            mask_score_list.append(score_exp_i)

            z_mask_list.append(z_mask) # Should be (B, d_z)

            if self.ablation_parameters.concept_matching:
                concept_selections, cs_inds = self.select_concepts(z_mask)
                all_concepts.append(concept_selections)
                cs_inds_list.append(cs_inds)
            all_concept_scores.append(score_exp_i)
        
        # Aggregate concepts by weighted summation:
        score_tensor = torch.cat(all_concept_scores, dim=-1).softmax(dim=-1).unsqueeze(1) # Now (B, 1, N_e)

        # Change below here for self-attention pooling

        if self.ablation_parameters.concept_matching:
            all_concepts_tensor = torch.stack(all_concepts, dim = 1) # Now (B, N_e, d_z)
        else:
            all_concepts_tensor = torch.stack(z_mask_list, dim = 1) # Now (B, N_e, d_z)

        agg_z_c = torch.bmm(score_tensor, all_concepts_tensor).squeeze()
        #print('agg_z_c', agg_z_c.shape)

        total_out_dict = {
            'pred': pred_regular, # Prediction on regular embedding (prediction branch)
            'pred_mask': pred_mask, # Prediction on masked embedding
            'mask_logits': torch.stack(mask_in_list, dim = -1), # Mask logits, i.e. before reparameterization + ste
            'concept_scores': score_tensor,
            'ste_mask': torch.stack(ste_mask_list, dim=-1),
            'smooth_src': smooth_src_list,
            'p': torch.cat(p_list, dim=-1),
            'all_z': (z_main, agg_z_c),
            'z_mask_list': torch.stack(z_mask_list, dim = -1),
            'concept_selections_inds': cs_inds_list
        }

        return total_out_dict

    def get_saliency_explanation(self, src, times, captum_input = False):
        '''
        Retrieves only saliency explanation (not concepts)
            - More efficient than calling forward due to less module calls
        '''

        z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        smooth_src_list, mask_in_list, ste_mask_list, p_list = [], [], [], []
        for i in range(self.n_explanations):
            smooth_src, mask_in, ste_mask, p = self.mask_generators[i](z_seq, src, times, get_tilde_mask = False)
            smooth_src_list.append(smooth_src)
            mask_in_list.append(mask_in)
            ste_mask_list.append(ste_mask)
            p_list.append(p)

        out_dict = {
            'smooth_src': smooth_src_list,
            'mask_in': mask_in_list,
            'ste_mask': ste_mask_list,
            'p': p_list
        }

        return out_dict


    def select_concepts(self, mask_z):
        '''
        - Performs concept selection based on predicted mask_z
        - Only works for one mask at a time - should be called in a loop
        mask_mu: (B, d_z)
        concepts: (N_c, 2, d_z)
        '''
        # MMD to all prototype means:

        B = mask_z.shape[0]

        # Need a probability map with (B, N_c)

        mz = mask_z.unsqueeze(1).repeat(1, self.n_concepts, 1) # (B, N_c, d_z)
        cm = self.concept_dists.unsqueeze(0).repeat(B, 1, 1) # (B, N_c, d_z)
        
        logit_map = F.cosine_similarity(mz, cm, dim=-1) # Reduce last dimension (d_z)
        log_prob_map = F.log_softmax(logit_map, dim = -1) # Apply along concept dimension
        
        # Prob map: (B, N_c)
        if self.training:
            concept_selections = F.gumbel_softmax(log_prob_map, hard = True) # Reparameterization selection - still (B, N_c)
        else:
            cs_inds = log_prob_map.argmax(dim=1)
            # Argmax-to-onehot conversion (cite: https://discuss.pytorch.org/t/how-to-convert-argmax-result-to-an-one-hot-matrix/125508)
            concept_selections = torch.zeros_like(log_prob_map).scatter_(1, cs_inds.unsqueeze(1), 1.0)
        #concept_selections = concept_selections.unsqueeze(-1).repeat(1,1,self.concept_dists.shape[-1]) # (B, N_c, d_z)

        # Multiply into selected concepts (multiplication performs selection):
        # (B, 1, N_c) @ (B, N_c, d_z) -> (B, 1, dz) -> (squeeze) (B, d_z)
        concepts = torch.bmm(concept_selections.unsqueeze(1), cm)  
        concepts = concepts.squeeze(1)

        return concepts, concept_selections

    def compute_loss(self, output_dict):

        # 1. Loss over size of mask
        if self.use_window:
            # print('mask_logits', output_dict['mask_logits'].shape)
            # exit()
            mask_loss = self.sizemaskloss(output_dict['mask_logits']) 
        else:
            mask_loss = self.loss_weight_dict['gsat'] * self.gsat_loss_fn(output_dict['mask_logits']) + self.loss_weight_dict['connect'] * self.connected_loss(output_dict['mask_logits'])

        # Concept decorrelation:

        # 2. Entropy on concept scores:
        loss_ent = 0.0 * self.entloss(output_dict['concept_scores'].squeeze(1))

        # 3. Smoothing loss (size of p):
        if self.ablation_parameters.trend_smoother_loss:
            loss_p = self.psizeloss(output_dict['p'])
        else:
            loss_p = 0.0 * self.psizeloss(output_dict['p'])
        #loss_p = 

        # 4. Decorrelation loss:
        #print('zmask', output_dict['z_mask_list'].shape)
        # loss_cor = F.cosine_similarity(output_dict['z_mask_list'][...,0], output_dict['z_mask_list'][...,1], dim = -1).mean()
        # loss_cor = 0.0 * 0.25 * (loss_cor + 1) / 2.0 # Manually normalize to [0,1]
        loss_cor = 0.0

        # 3. Contrastive loss between concept embedding and actual embedding
        # Relegate to trainer? - YES

        return {'mask_loss': mask_loss, 'ent_loss': loss_ent, 'p_loss': loss_p, 'corr_loss': loss_cor}

    def entloss(self, probs):
        # Probs should be (B, N_c)
        return -1.0 * (torch.log(probs + 1e-6) * probs).sum(dim=-1).mean() # Return scalar

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)


    @torch.no_grad()
    def find_closest_for_prototypes(self, src, times, captum_input = False):
        '''
        Only used for evaluation
        '''
        self.eval()

        out_dict = self.forward(src, times, captum_input = False)

        B, d, Ne = out_dict['z_mask_list'].shape
        to_search = out_dict['z_mask_list'].view(B + Ne, -1)
        mask_logits = out_dict['mask_logits']
        print('mask logits', mask_logits.shape)
        masks = torch.cat([mask_logits[...,0], mask_logits[...,1]], dim = 0)
        smooth_src = torch.cat(out_dict['smooth_src'], dim = 1)
        print('smooth_src', smooth_src.shape)

        found_masks = []
        found_smooth_src = []
        for c in range(self.concept_dists.shape[0]):
            ci = self.concept_dists[c,:].unsqueeze(0).repeat(to_search.shape[0],1)

            # Search by cosine sim:
            sims = F.cosine_similarity(ci, to_search, dim = -1)
            best_i = sims.argmin()

            found_masks.append(masks[best_i,:])
            found_smooth_src.append(smooth_src[:,best_i,:])

        return found_masks, found_smooth_src


    def set_config(self): # TODO: update
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'n_concepts': self.n_concepts,
            'n_explanations': self.n_explanations,
            'gsat_r': self.gsat_r,
            'transformer_args': self.transformer_args,
            'trend_smoother': self.trend_smoother,
            'use_window': self.use_window,
            'ablation_parameters': self.ablation_parameters,
        }