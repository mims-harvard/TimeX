import torch
from torch import nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.functional import mahalanobis_dist
from txai.models.mask_generators.gumbel import GumbelMask

class Dataset_Z(torch.utils.data.Dataset):
    def __init__(self, z_emb, labels):
        self.z_emb = z_emb
        self.labels = labels

    def __getitem__(self, idx):
        return self.z_emb[idx], self.labels[idx]

    def __len__(self):
        return self.z_emb.shape[0]

def train_concept_scorer(model, z_emb, batch_id, epochs = 100, batch_size = 32):

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01)

    # Break into val and train:
    np_labels = batch_id.detach().cpu().numpy()
    train_inds, val_inds = train_test_split(np.arange(z_emb.shape[0]), test_size = 0.1, stratify = np_labels)
    train_inds, val_inds = torch.from_numpy(train_inds).long(), torch.from_numpy(val_inds).long()

    z_emb_train, z_emb_val = z_emb[train_inds,:], z_emb[val_inds,:]

    yval = np_labels[train_inds.cpu().numpy()]

    loader = torch.utils.data.DataLoader(Dataset_Z(z_emb_train, batch_id[train_inds]), batch_size = batch_size, shuffle = True)

    sdict_best, val_best = None, -1e9

    for epoch in range(epochs):
        for z, y in loader:
            optimizer.zero_grad()
            out = model(z)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_out = model(z_emb_val)
        acc = (val_out.argmax(dim=1) == batch_id[val_inds]).float().mean().item()

        if acc > val_best:
            sdict_best = model.state_dict()
            val_best = acc

        if epoch % 10 == 0:
        # TODO: Validation
            print('Val acc. = {:.4f}'.format(acc))

    model.load_state_dict(sdict_best)


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
            smooth_concepts = False,
            distance_method = 'mahalanobis',
            n_masks = 2,
            ):
        super(CBMv1, self).__init__()
        
        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        # n_masks
        # dmethod
        self.type_masktoken = type_masktoken
        self.type_archmask = type_archmask
        self.mask_seed = mask_seed
        self.smooth_concepts = smooth_concepts

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
            GumbelMask(self.d_inp + d_pe, self.d_inp, max_len, type_masktoken = type_masktoken, type_archmask = type_archmask, masktoken_kwargs = masktoken_kwargs, seed = mask_seed, smooth_concepts = smooth_concepts),
            GumbelMask(self.d_inp + d_pe, self.d_inp, max_len, type_masktoken = type_masktoken, type_archmask = type_archmask, masktoken_kwargs = masktoken_kwargs, seed = mask_seed, smooth_concepts = smooth_concepts),
        ])
        self.n_masks = n_masks # Hardcoded, can fix later

        # Fix at 4 outputs for now, can scale later
        self.predictor = nn.Linear(4, self.n_classes, bias = False)

        self.distance_method = distance_method

        if self.distance_method == 'mlp':
            # TODO: Non-hard coding of intermediate embedding sizes
            self.distance_mlp = nn.Sequential(
                torch.nn.Linear(self.d_pe + self.d_inp, 128), 
                torch.nn.PReLU(), 
                torch.nn.Linear(128, 128), 
                torch.nn.PReLU(), 
                torch.nn.Linear(128, self.n_concepts)
            )

        self.set_config()
        self.frozen_mlp = False

    def store_concept_bank(self, src, times, batch_id):
        '''
        Stores a bank of concepts in the local module
        '''
        self.eval()
        if self.distance_method == 'mahalanobis':
            with torch.no_grad():
                self.mu = []
                self.sigma_inv = []
                for ui in torch.unique(batch_id):
                    inmask = (batch_id == ui)
                    emb = self.encoder.embed(src[:,inmask,:], times[:,inmask], aggregate = True)
                    self.mu.append(emb.mean(dim=0))
                    self.sigma_inv.append(torch.linalg.pinv(torch.cov(emb.T)))
        elif self.distance_method == 'mlp' and not self.frozen_mlp:
            # Train basic MLP to learn concept embeddings:
            with torch.no_grad():
                z_concepts =  self.encoder.embed(src, times, aggregate = True)

            train_concept_scorer(self.distance_mlp, z_concepts, batch_id)
        elif self.distance_method == 'centroid':
            with torch.no_grad():
                self.mu = []
                for ui in torch.unique(batch_id):
                    inmask = (batch_id == ui)
                    emb = self.encoder.embed(src[:,inmask,:], times[:,inmask], aggregate = True)
                    self.mu.append(emb.mean(dim=0))

        self.train()

    def load_concept_bank(self, path):
        d = torch.load(path)
        self.mu, self.sigma_inv = d['mu'], d['sigma_inv']

    def freeze_encoder(self, freeze_mlp = True):

        for param in self.encoder.parameters():
            param.requires_grad = False

        if freeze_mlp:
            self.freeze_mlp_dist()

    def freeze_mlp_dist(self):
        if self.distance_method == 'mlp':
            for param in self.distance_mlp.parameters():
                param.requires_grad = False
            #self.frozen_mlp = True

    def unfreeze_encoder(self):

         for param in self.encoder.parameters():
            param.requires_grad = True

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

        if self.frozen_mlp or self.distance_method == 'mlp':
            self.freeze_mlp_dist()

        ma_scores = []
        mask_list = []
        logits_list = []
        for n in range(self.n_masks):
            # Through mask layers
            #print('src', src.shape)
            masked_src, masked_times, mask, logits, attn_mask = self.mask_gens[n](src, times, enc, captum_input = False)
            mask_list.append(mask)
            logits_list.append(logits)

            # Mask back through encoder
            mask_enc = self.encoder.embed(masked_src, masked_times, given_attn_mask = attn_mask, captum_input = False, aggregate = True)

            # Calc distances:
            if self.distance_method == 'mahalanobis':
                md_list = []
                for c in range(self.n_concepts):
                    md_c = mahalanobis_dist(mask_enc, self.mu[c], self.sigma_inv[c])
                    md_list.append(md_c)

                md_agg = torch.cat(md_list, dim=-1).squeeze()

            elif self.distance_method == 'mlp':
                md_agg = self.distance_mlp(mask_enc)

            elif self.distance_method == 'centroid':
                md_list = []
                for c in range(self.n_concepts):
                    md_list.append(torch.nn.functional.cosine_similarity(mask_enc, self.mu[c], dim = 1).unsqueeze(-1))

                md_agg = torch.cat(md_list, dim=-1)

            # Softmax scores:
            #print('md agg', md_agg.shape)
            md_agg = md_agg.softmax(dim=1)
            ma_scores.append(md_agg)

        # Concat all scores:
        concept_scores = torch.cat(ma_scores, dim=-1)

        # Through mlp
        yhat = self.predictor(concept_scores)
        #print('yhat', yhat.shape)

        if self.distance_method == 'mlp':
            for param in self.distance_mlp.parameters():
                param.requires_grad = True

        # Return: prediction, concept scores, masks, logits
        return yhat, concept_scores, mask_list, logits_list


    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder.d_inp,
            'max_len': self.encoder.max_len,
            'n_classes': self.encoder.n_classes,
            'n_concepts': self.n_concepts,
            'type_masktoken': self.type_masktoken,
            'type_archmask': self.type_archmask,
            'mask_seed': self.mask_seed,
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
            'smooth_concepts': self.smooth_concepts,
            'distance_method': self.distance_method,
            'n_masks': self.n_masks
        }