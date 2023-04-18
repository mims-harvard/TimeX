import torch
from torch import nn
import torch.nn.functional as F
from torch import linalg

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.modelv2 import MaskGenStochasticDecoder

from txai.utils.predictors.loss import GSATLoss

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


class Modelv6Variational(nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_concepts,
            n_explanations,
            gsat_r, 
            transformer_args = transformer_default_args,
            trend_smoother = True,
        ):
        super(Modelv6Univariate, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = transformer_default_args['d_pe']
        self.n_classes = n_classes
        self.trend_smoother = trend_smoother
        self.transformer_args = transformer_args
        self.n_concepts = n_concepts
        self.n_explanations = n_explanations
        self.gsat_r = gsat_r

        d_z = (self.d_inp + self.d_pe)
        
        self.encoder_main = TransformerMVTS(
            d_inp = d_inp,  # Dimension of input from samples (must be constant)
            max_len = max_len, # Max length of any sample to be fed into model
            n_classes = self.n_classes, # Number of classes for classification head
            **self.transformer_args
        )

        self.main_mu_net  = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, d_z))
        self.main_logvar_net = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, d_z))

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

        self.mask_mu_net  = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, d_z))
        self.mask_logvar_net = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, d_z))
        self.mask_score_net = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, 1))

        # For decoder, first value [0] is actual value, [1] is mask value (predicted logit)
        self.mask_decoder_net = torch.Sequential(torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(d_z, d_z), torch.nn.PReLU(), torch.nn.Linear(2))

        self.mask_generators = nn.ModuleList()
        for _ in range(self.n_explanations):
            mgen = MaskGenStochasticDecoder(d_z = (self.d_inp + self.d_pe), max_len = max_len, trend_smoother = trend_smoother)
            self.mask_generators.append(mgen)

        # Below is a sequence-level embedding - (N_c, T, 2, d_z) -> T deals with the length of the time series
        #   - The above is done to allow for sequence-level decoding
        self.concept_seq_dists = nn.Parameter(torch.randn(self.n_concepts, self.max_len, 2, d_z)) # 2 defines mu and logvar

        # Setup loss functions:
        self.gsat_loss_fn = GSATLoss(r = self.gsat_r)

        self.set_config()

    def forward(self, src, times, captum_input = False):
        
        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)

        pred_regular, z_main_pre_reparam, z_seq_main = self.encoder_main(src, times, captum_input = False, get_agg_embed = True)
        z_seq = self.encoder_pret.embed(src, times, captum_input = False, aggregate = False)

        main_mu = self.main_mu_net(z_main_pre_reparam)
        main_logvar= self.main_logvar_net(z_main_pre_reparam)
        z_main = self.reparameterize(main_mu, main_logvar)

        # Generate smooth_src: # TODO: expand to lists
        smooth_src_list, mask_in_list, ste_mask_list, smoother_stats_list = [], [], [], []
        pred_mask_list, z_mask_list = [], []
        mask_dec_list, mask_score_list = [], []
        all_concepts = []
        all_concept_scores = []
        for i in range(self.n_explanations): # Iterate over all explanations
            smooth_src, mask_in, ste_mask, smoother_stats = self.mask_generators[i](z_seq, src, times, get_tilde_mask = False)
            smooth_src_list.append(smooth_src)
            mask_in_list.append(mask_in)
            ste_mask_list.append(transform_to_attn_mask(ste_mask))
            smoother_stats_list.append(smoother_stats)

            pred_mask, z_mask_pre_reparam, z_seq_mask = self.encoder_t(smooth_src, times, attn_mask = ste_mask, get_agg_embed = True)
            pred_mask_list.append(pred_mask)

            # Variational inference:
            mask_mu = self.mask_mu_net(z_mask_pre_reparam)
            mask_logvar = self.mask_logvar_net(z_mask_pre_reparam)
            z_mask = self.reparameterize(mask_mu, mask_logvar)

            # Gives score of current explanation considered:
            score_exp_i = self.mask_score_net(z_mask_pre_reparam) # Not trained on variational for now
            # Above should be (B,1)
            mask_dec = self.mask_decoder_net(z_mask)
            
            mask_score_list.append(score_exp_i)
            mask_dec_list.append(mask_dec) # TODO: convert to summation of losses

            z_mask_list.append(z_mask)

            # TODO: Perform prototype matching:
            concept_selections, concept_mu_i, concept_logvar_i = self.select_concepts(mask_mu)

            all_concepts.append(concepts_selections)
            all_concept_scores.append(score_exp_i)
        
        # Aggregate concepts by weighted summation:
        score_tensor = torch.cat(all_concept_scores, dim=-1).softmax(dim=-1).unsqueeze(1) # Now (B, 1, N_e)
        all_concepts_tensor = torch.stack(all_concepts, dim = 1) # Now (B, N_e, d_z)

        agg_z_c = torch.bmm(score_tensor, all_concepts_tensor)

        total_out_dict = {
            'pred': pred_regular,
            'pred_mask': pred_mask,
            'mask_logits': mask_in_list,
            'concept_scores': score_tensor,
            'agg_z_c': agg_z_c,
        }

        # if self.training:
        #     total_loss = self.compute_loss(total_out_dict)

        # Return is dictionary:
        d = {
            'pred': pred_regular,
            'pred_mask': pred_mask,
            'mask_logits': mask_in,
            'ste_mask': ste_mask,
            'smoother_stats': smoother_stats, 
            'smooth_src': smooth_src,
            'all_preds': (pred_regular, pred_mask),
            'all_z': (z_main, z_mask)
        }

        return d

    def select_concepts(self, mask_mu):
        '''
        - Performs concept selection based on predicted mu
        mask_mu: (B, d_z)
        concepts: (N_c, 2, d_z)
        '''
        # MMD to all prototype means:

        B = mask_mu.shape[0]

        cdists_max_mu = self.concept_dists[:,:,0,:].max(dim = 1)
        cdists_max_logvar = self.concept_dists[:,:,1,:].max(dim = 1) # Aggregate along time dimension
        # Both should now be (N_c, d_z)

        # Need a probability map with (B, N_c)

        mm = mask_mu.unsqueeze(1).repeat(1, self.n_concepts, 1) # (B, N_c, d_z)
        cm = cdists_max_mu.unsqueeze(0).repeat(B, 1, 1) # (B, N_c, d_z)
        
        logit_map = linalg.norm(mm - cm, ord=2, dim=-1) # Reduce last dimension (d_z)
        log_prob_map = F.log_softmax(logit_map, dim = -1) # Apply along concept dimension
        
        # Prob map: (B, N_c)
        concept_selections = F.gumbel_softmax(log_prob_map, hard = True) # Reparameterization selection - still (B, N_c)
        #concept_selections = concept_selections.unsqueeze(-1).repeat(1,1,self.concept_dists.shape[-1]) # (B, N_c, d_z)

        # Multiply into selected concepts (multiplication performs selection):
        # (B, 1, N_c) @ (B, N_c, d_z) -> (B, 1, dz) -> (B, d_z) (for both mu and logvar)
        concepts_mu_pre_reparam = torch.bmm(concept_selctions.unsqueeze(1), cdists_max_mu.unsqueeze(0).repeat(B,1,1))  
        concepts_mu_pre_reparam = concepts_pre_reparam.squeeze(1)

        concepts_logvar_pre_reparam = torch.bmm(concept_selctions.unsqueeze(1), cdists_max_logvar.unsqueeze(0).repeat(B,1,1))  
        concepts_logvar_pre_reparam = concepts_pre_reparam.squeeze(1)
        # Should now have: (B, d_z) for each

        # Reparameterize each concept:
        concepts = self.reparameterize(concepts_mu_pre_reparam, concepts_logvar_pre_reparam)

        # Selected concepts on sequence level:
        

        return concepts, concepts_mu_pre_reparam, concepts_logvar_pre_reparam

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_loss(self, src, times, output_dict):

        # 1. Loss GSAT
        mask_loss = self.gsat_loss_fn(output_dict['mask_logits']) # Loss is pointwise, so can apply to all at once

        # 2. KL Loss for mask embeddings
        kl_mask = self.klloss()

        # 3. KL loss for concept embeddings used
        kl_concept = self.klloss()

        # 4. Reconstruction loss of transformed embeddings
        rec_x, rec_mask = self.masked_reconstruction_loss(
            recon_masked_x = None,
            src = src,
            ste_mask = output_dict['ste_mask'],
            mask_logits = output_dict['mask_logits']
        )

        # 5. Entropy on concept scores:
        loss_ent = self.entloss(output_dict['concept_scores'].squeeze(1))

        # 6. Contrastive loss between concept embedding and actual embedding
        # Relegate to trainer?

    # Implement loss functions:
    def klloss(self, mu, logvar): # TODO: loss function implementation
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - (mu ** 2) - logvar.exp(), dim=1), dim = 0)
        return kld

    def entloss(self, probs):
        # Probs should be (B, N_c)
        return -1.0 * (torch.log(probs + 1e-6) * probs).sum(dim=-1).mean() # Return scalar

    def masked_reconstruction_loss(self, recon_masked_x, src, ste_mask, mask_logits):
        '''
        Input shapes:
            recon_masked_x: (N_c, B, T, 2)
            src: (T, B, d)
            ste_mask: (N_c, B, T, 1)
            mask_logits: (N_c, B, T, 1)
        '''
        # Reconstruction error between reconstructed masked x and masked_x

        # Only penalize the regions where we actually mask-in the src
        # Need ste_mask here because we want to *completely ignore* those areas we masked-out
        rec_x = (torch.sqrt(((recon_masked_x[...,0].unsqueeze(-1) - src) ** 2)) * ste_mask).sum() / (ste_mask.sum())
        
        rec_mask = (torch.sqrt((recon_masked_x[...,1].unsqueeze(-1) - mask_logits) ** 2)).mean() # Need to reconstruct entire mask

        return rec_x, rec_mask

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self): # TODO: update
        self.config = {
            'd_inp': self.encoder_main.d_inp,
            'max_len': self.max_len,
            'n_classes': self.encoder_main.n_classes,
            'transformer_args': self.transformer_args,
            'trend_smoother': self.trend_smoother,
        }