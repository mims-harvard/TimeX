import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from txai.models.positional_enc import PositionalEncodingTF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerEncPhi(nn.Module):

    def __init__(self,
            d_inp,
            max_len,
            d_pe,
            nhead = 1,
            dim_feedforward = 72,
            dropout = 0.25,
            nlayers = 1,
            MAX = 10000,
            batch_first = False
            ):
        
        super(TransformerEncPhi, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.nlayers = nlayers
        self.batch_first = batch_first
        
        # Set up Transformer encoder layer 1 (to learn extractions):
        encoder_layers = TransformerEncoderLayer(
            d_model = self.d_pe + self.d_inp, 
            nhead = self.nhead, 
            dim_feedforward = self.dim_feedforward, 
            dropout = self.dropout,
            batch_first = self.batch_first)
        self.encoder = TransformerEncoder(encoder_layers, self.nlayers)

        # Positional encodings:
        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        self.d_out = self.d_pe + self.d_inp

    def forward(self, src, times, static = None, captum_input = False):
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

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
        output = self.encoder(x, src_key_padding_mask = overlen_mask)

        return output.transpose(0,1), overlen_mask

class TransformerConvSNEncPhi(nn.Module):

    def __init__(self,
            d_inp,
            max_len,
            d_pe = 16,
            nhead = 1,
            dim_feedforward = 72,
            dropout = 0.25,
            nlayers = 1,
            MAX = 10000,
            batch_first = False,
            snet_layers = 2,
            snet_channels = 32,
            snet_kernelsize = 5,
            ):

        super(TransformerConvSNEncPhi, self).__init__()

        self.d_inp = d_inp
        self.max_len = max_len
        self.d_pe = d_pe
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.nlayers = nlayers
        self.batch_first = batch_first

        # Sensornet hyperparams:
        self.snet_layers = snet_layers
        self.snet_channels = snet_channels
        self.snet_kernelsize = snet_kernelsize
        
        # Set up Transformer encoder layer 1 (to learn extractions):
        encoder_layers = TransformerEncoderLayer(
            d_model = self.d_pe + self.d_inp, 
            nhead = self.nhead, 
            dim_feedforward = self.dim_feedforward, 
            dropout = self.dropout,
            batch_first = self.batch_first)
        self.encoder = TransformerEncoder(encoder_layers, self.nlayers)

        # Positional encodings:
        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        # Convolutional SensorNet:
        # Create sensor net, i.e. flat-kernel CNN over each sensor:
        modules = []
        for l in range(self.snet_layers):
            if l == 0:
                c = torch.nn.Sequential(
                    torch.nn.Conv2d(1, self.snet_channels, kernel_size = (self.snet_kernelsize, 1), padding = 'same'),
                    torch.nn.BatchNorm2d(self.snet_channels, track_running_stats = False),
                    torch.nn.PReLU(),
                )
            else:
                c = torch.nn.Sequential(
                    torch.nn.Conv2d(self.snet_channels, self.snet_channels, kernel_size = (self.snet_kernelsize, 1), padding = 'same'),
                    torch.nn.BatchNorm2d(self.snet_channels, track_running_stats = False),
                    torch.nn.PReLU(),
                )

            modules.append(c)

        modules.append(torch.nn.Conv2d(self.snet_channels, 1, kernel_size = (1,1), padding = 'same'))

        self.snet = torch.nn.Sequential(*modules)

        # Set output shape:
        self.d_out = self.d_pe + self.d_inp + self.d_inp 

    def forward(self, src, times, static = None, captum_input = False):
        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            if captum_input:
                src = src.unsqueeze(dim=0)
            else:
                src = src.unsqueeze(dim=1)

        lengths = torch.sum(times > 0, dim=0)
        maxlen, batch_size = src.shape[0], src.shape[1]

        pe = self.pos_encoder(times).to(device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        # mask out the all-zero rows
        overlen_mask = ( torch.arange(maxlen).unsqueeze(0) >= (lengths.cpu().unsqueeze(-1)) ).to(device)

        if overlen_mask.dim() == 1:
            # Unsqueeze if only using one example (must have B first in dim.)
            overlen_mask = overlen_mask.unsqueeze(dim=1)

        # Transformer must have (T, B, d)
        # Mask is (B, T)
        output = self.encoder(x, src_key_padding_mask = overlen_mask)

        # Concatenate sensor outputs to output:

        # Run through sensornet:
        # Flip batch back to first dim
        snetout = self.snet(src.transpose(0,1).unsqueeze(1)).squeeze(1)

        output = torch.cat([output.transpose(0,1), snetout], dim=-1)

        return output, overlen_mask

if __name__ == '__main__':
    # Testing for transformer:

    model = TransformerConvSNEncPhi(
        d_inp = 4,
        max_len = 50,
        d_pe = 16,
        nhead = 1,
        dim_feedforward = 72,
        dropout = 0.25,
        nlayers = 1,
        MAX = 10000,
        batch_first = False,
        snet_layers = 2,
        snet_channels = 32,
        snet_kernelsize = 5,
    )

    model.to(device)

    # Random input:
    X = torch.randn(64, 50, 4).cuda()
    times = torch.arange(50).unsqueeze(0).repeat(64,1).cuda()
    out, _ = model(X, times, captum_input = True)

    print(out.shape)
