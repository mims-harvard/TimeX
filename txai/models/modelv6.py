import torch
from torch import nn
import torch.nn.functional as F
from torch import linalg

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.modelv2 import MaskGenStochasticDecoder_NoCycleParam
from txai.models.mask_generators.window_mask import MaskGenWindow

from txai.utils.predictors.loss import GSATLoss
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



class OptimizerHelper:

    def __init__(self, 
            model,
        ):

        self.clf_optimizer = torch.optim.AdamW()
        self.T_optimizer = None
        self.scorer_optimizer = None

    def zero_grad(self):
        pass

    def step(self):
        pass

class Modelv6(nn.Module):
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
        ):
        super(Modelv6, self).__init__()

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
            mgen = MaskGenWindow(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother)
            self.mask_generators.append(mgen)

        # Below is a sequence-level embedding - (N_c, T, 2, d_z) -> T deals with the length of the time series
        #   - The above is done to allow for sequence-level decoding
        self.concept_dists = nn.Parameter(torch.randn(self.n_concepts, d_z)) # 2 defines mu and logvar

        # Bilinear decoder for contrastive scoring:
        self.embedding_scorer = nn.Bilinear((self.d_inp + self.d_pe), (self.d_inp + self.d_pe), out_features = 1)

        # Setup loss functions:
        #self.gsat_loss_fn = GSATLoss(r = self.gsat_r)
        self.psizeloss = PSizeLoss()
        self.sizemaskloss = SizeMaskLoss(mean = True, target_val = self.size_mask_target_val)
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
        mask_sides_list = []
        all_concepts = []
        all_concept_scores = []
        for i in range(self.n_explanations): # Iterate over all explanations
            smooth_src, mask_in, ste_mask, p, mask_sides = self.mask_generators[i](z_seq, src, times, get_tilde_mask = False)
            ste_mask_attn = transform_to_attn_mask(ste_mask)

            smooth_src_list.append(smooth_src)
            mask_in_list.append(mask_in)
            ste_mask_list.append(ste_mask)
            p_list.append(p)
            mask_sides_list.append(mask_sides)

            pred_mask, z_mask, z_seq_mask = self.encoder_t(smooth_src, times, attn_mask = ste_mask_attn, get_agg_embed = True)
            pred_mask_list.append(pred_mask)

            # Gives score of current explanation considered:
            score_exp_i = self.mask_score_net(z_mask) # Not trained on variational for now
            # Above should be (B,1)
            
            mask_score_list.append(score_exp_i)

            z_mask_list.append(z_mask) # Should be (B, d_z)

            # TODO: Perform prototype matching:
            #concept_selections = self.select_concepts(z_mask) UNMASK WHEN USING CONCEPTS

            #all_concepts.append(concept_selections)
            all_concept_scores.append(score_exp_i)
        
        # Aggregate concepts by weighted summation:
        score_tensor = torch.cat(all_concept_scores, dim=-1).softmax(dim=-1).unsqueeze(1) # Now (B, 1, N_e)
        #all_concepts_tensor = torch.stack(all_concepts, dim = 1) # Now (B, N_e, d_z)
        all_concepts_tensor = torch.stack(z_mask_list, dim = 1) # Now (B, N_e, d_z)

        agg_z_c = torch.bmm(score_tensor, all_concepts_tensor).squeeze()
        #print('agg_z_c', agg_z_c.shape)

        total_out_dict = {
            'pred': pred_regular, # Prediction on regular embedding (prediction branch)
            'pred_mask': pred_mask, # Prediction on masked embedding
            'mask_logits': torch.stack(mask_in_list, dim = -1), # Mask logits, i.e. before reparameterization + ste
            'mask_components': mask_sides_list, # Original distributions
            'concept_scores': score_tensor,
            'ste_mask': torch.stack(ste_mask_list, dim=-1),
            'smooth_src': smooth_src_list,
            'p': torch.cat(p_list, dim=-1),
            'all_z': (z_main, agg_z_c),
            'z_mask_list': torch.cat(z_mask_list, dim = 0)
        }

        #print('m logits', total_out_dict['mask_logits'].shape)

        # if self.training:
        #     total_loss = self.compute_loss(total_out_dict)

        # Return is dictionary:
        # d = {
        #     'pred': pred_regular,
        #     'pred_mask': pred_mask,
        #     'mask_logits': mask_in,
        #     'ste_mask': ste_mask,
        #     'p': p, 
        #     'smooth_src': smooth_src,
        #     'all_preds': (pred_regular, pred_mask),
        #     'all_z': (z_main, z_mask)
        # }

        return total_out_dict

    def score_contrastive(self, target_embeddings, concept_embeddings):
        # print('targ', target_embeddings.shape)
        # print('con', concept_embeddings.shape)

        # Need to bound:
        scores = self.embedding_scorer(target_embeddings, concept_embeddings).tanh()
        #print('score', scores.shape)
        return scores

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
        concept_selections = F.gumbel_softmax(log_prob_map, hard = True) # Reparameterization selection - still (B, N_c)
        #concept_selections = concept_selections.unsqueeze(-1).repeat(1,1,self.concept_dists.shape[-1]) # (B, N_c, d_z)

        # Multiply into selected concepts (multiplication performs selection):
        # (B, 1, N_c) @ (B, N_c, d_z) -> (B, 1, dz) -> (squeeze) (B, d_z)
        concepts = torch.bmm(concept_selections.unsqueeze(1), cm)  
        concepts = concepts.squeeze(1)

        return concepts

    def compute_loss(self, output_dict):

        # 1. Loss over size of mask

        #mside_list = output_dict['mask_components']
        #mask_loss = self.sizemaskloss(*mside_list[0]) + self.sizemaskloss(*mside_list[1])
        # for i in range(len(mside_list)):
        #     if i == 0:
        #         mask_loss = self.sizemaskloss(*mside_list[i])
        #     else:
        #         mask_loss += self.sizemaskloss(*mside_list[i])

        #mask_loss = mask_loss / len(mside_list) # Normalize by number of explanations

        mask_loss = self.sizemaskloss(output_dict['mask_logits']) # Loss is pointwise, so can apply to all at once

        # Concept decorrelation:

        # 2. Entropy on concept scores:
        loss_ent = 0.0 * self.entloss(output_dict['concept_scores'].squeeze(1))
        #loss_ent = 0

        # 3. Smoothing loss (size of p):
        # print('smoother_ss', output_dict['p'])
        # exit()
        #loss_p = self.psizeloss(output_dict['p'])
        loss_p = 0

        # 3. Contrastive loss between concept embedding and actual embedding
        # Relegate to trainer? - YES

        return {'mask_loss': mask_loss, 'ent_loss': loss_ent, 'p_loss': loss_p}

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

        to_search = out_dict['z_mask_list']
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
        }