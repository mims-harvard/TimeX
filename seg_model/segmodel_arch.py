import torch
from torch import nn

from txai.models.encoders.positional_enc import PositionalEncodingTF 


class SegTransformer(nn.Module):
    '''
    Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''

    def __init__(self, 
            d_inp: int, 
            d_model: int = 64, 
            nhead: int = 4, 
            nlayers: int = 3, 
            dropout: float = 0.5,
            n_concept = 4,
        ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncodingTF(d_model, max_len=500)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, time: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(time)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output