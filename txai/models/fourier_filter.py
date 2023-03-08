import torch
from torch import nn

from txai.models.fourier_layers import FourierMasker
from txai.models.encoders.transformer_simple import TransformerMVTS

class FourierFilterModel(nn.Module):
    def __init__(self,
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            enc_dropout = None, # Encoder dropout 
            nhead = 1, # Number of attention heads
            trans_dim_feedforward = 72, # Number of hidden layers 
            trans_dropout = 0.25, # Dropout rate in Transformer encoder
            nlayers = 1, # Number of Transformer layers
            aggreg = 'mean', # Aggregation of transformer embeddings
            MAX = 10000, # Arbitrary large number
            norm_embedding = True,
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
    ):
        super(FourierFilterModel, self).__init__()
        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.n_classes = n_classes

        self.fourier_scorer = FourierMasker(maxlen = max_len)

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

        self.set_config()

    def forward(self, src, times, captum_input = False):

        if captum_input:
            src = src.transpose(0, 1)
            times = times.transpose(0, 1)
        
        new_src, logits, masks = self.fourier_scorer(src)

        out = self.encoder(new_src, times, captum_input = False)

        return out, new_src, masks, logits

    def save_state(self, path):
        tosave = (self.state_dict(), self.config)
        torch.save(tosave, path)

    def set_config(self):
        self.config = {
            'd_inp': self.encoder.d_inp,
            'max_len': self.encoder.max_len,
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
        }
