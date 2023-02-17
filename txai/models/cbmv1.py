import torch
from torch import nn

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.functional import mahalanobis_dist
from txai.models.mask_generators.gumbel import GumbelMask

class CBMv1(torch.nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            n_concepts = 2,
            type_masktoken = 'dyna_norm_datawide',
            type_archmask = None,
            masktoken_kwargs = {},
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
            ):
        super(CBMv1, self).__init__()
        
        self.d_inp = d_inp
        self.n_classes = n_classes
        self.n_concepts = n_concepts

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

        # Fix at two for now:
        self.mask_gens = nn.ModuleList([
            GumbelMask(self.d_inp + d_pe, self.d_inp, max_len, type_masktoken = type_masktoken, type_archmask = type_archmask, masktoken_kwargs = masktoken_kwargs, seed = mask_seed),
            GumbelMask(self.d_inp + d_pe, self.d_inp, max_len, type_masktoken = type_masktoken, type_archmask = type_archmask, masktoken_kwargs = masktoken_kwargs, seed = mask_seed),
        ])
        self.n_masks = 2 # Hardcoded, can fix later

        # Fix at 4 outputs for now, can scale later
        self.predictor = nn.Linear(4, self.n_classes, bias = False)


    def store_concept_bank(self, src, times, batch_id):
        '''
        Stores a bank of concepts in the local module
        '''
        self.eval()
        with torch.no_grad():
            self.mu = []
            self.sigma_inv = []
            for ui in torch.unique(batch_id):
                inmask = (batch_id == ui)
                emb = self.encoder.embed(src[:,inmask,:], times[:,inmask], aggregate = True)
                self.mu.append(emb.mean(dim=0))
                self.sigma_inv.append(torch.linalg.pinv(torch.cov(emb.T)))
        self.train()

    def load_concept_bank(self, path):
        d = torch.load(path)
        self.mu, self.sigma_inv = d['mu'], d['sigma_inv']

    def forward(self, src, times, captum_input = False):
        '''
        In this model, represent concept bank as:
            (\mu_1, \Sigma_1), (\mu_2, \Sigma_2) with \mu and \Sigma being centroids 
                and covariance matrices of each concept dist 
        '''

        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)
        
        if (src.isnan().sum() > 0) or (times.isnan().sum() > 0):          
            print('X', src.isnan().sum())
            print('time', times.isnan().sum())
        enc = self.encoder.embed(src, times, captum_input = False, aggregate = False)
        if enc.isnan().sum() > 0:
            print('Enc', enc.isnan().sum() / (enc.shape[0] * enc.shape[1] * enc.shape[2]))
        # Is a seq2seq output

        ma_scores = []
        mask_list = []
        logits_list = []
        for n in range(self.n_masks):
            # Through mask layers
            masked_src, masked_times, mask, logits = self.mask_gens[n](src, times, enc, captum_input = False)
            mask_list.append(mask)
            logits_list.append(logits)

            # Mask back through encoder
            mask_enc = self.encoder.embed(masked_src, masked_times, captum_input = False, aggregate = True)

            # Calc distances:
            md_list = []
            for c in range(self.n_concepts):
                md_c = mahalanobis_dist(mask_enc, self.mu[c], self.sigma_inv[c])
                md_list.append(md_c)

            md_agg = torch.cat(md_list, dim=-1).squeeze()
            # Softmax scores:
            #print('md agg', md_agg.shape)
            md_agg = md_agg.softmax(dim=1)
            ma_scores.append(md_agg)

        # Concat all scores:
        concept_scores = torch.cat(ma_scores, dim=-1)

        # Through mlp
        yhat = self.predictor(concept_scores)
        #print('yhat', yhat.shape)

        # Return: prediction, concept scores, masks, logits
        return yhat, concept_scores, mask_list, logits_list

