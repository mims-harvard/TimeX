import torch
from txai.models.positional_enc import PositionalEncodingTF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DualAttentionTransformer(torch.nn.Module):

    def __init__(self,
        T,
        d,
        d_pe,
        nhead = 1,
        nlayers = 1,
        dim_feedforward = 72,
        dropout = 0.1,
        batch_first = False,
        cnn_kernel_len = 5,
        cnn_channels = 32):

        super(DualAttentionTransformer, self).__init__()

        self.T = T
        self.d = d
        self.d_pe = d_pe
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.nlayers = nlayers
        self.batch_first = batch_first
        self.cnn_kernel_len = cnn_kernel_len
        self.C = cnn_channels

        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model = self.d + self.d_pe, 
            nhead = self.nhead, 
            dim_feedforward = self.dim_feedforward, 
            dropout = self.dropout,
            batch_first = False)
        self.time_encoder = torch.nn.TransformerEncoder(encoder_layers, self.nlayers)

        self.sensor_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.C, kernel_size = (self.cnn_kernel_len, 1), padding = 'same'),
            torch.nn.BatchNorm2d(self.C, track_running_stats = False),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.C, self.C, kernel_size = (self.cnn_kernel_len + 2, 1), padding = 'same'),
            torch.nn.BatchNorm2d(self.C, track_running_stats = False),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.C, self.C, kernel_size = (self.cnn_kernel_len + 4, 1), padding = 'same'),
            torch.nn.BatchNorm2d(self.C, track_running_stats = False),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.C, 1, kernel_size = (1, 1)), # 1x1 CNN to pool down into 1 channel
            torch.nn.PReLU(),
        )

        # Calculate size of pooled-down:
        self.sensornet_out_size = (self.T - self.cnn_kernel_len + 1)
        self.sensornet_out_size = (self.sensornet_out_size - self.cnn_kernel_len - 2 + 1)
        self.sensornet_out_size = (self.sensornet_out_size - self.cnn_kernel_len - 4 + 1)
        #print('Calculated', self.sensornet_out_size)

        sensor_encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model = self.T,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward, 
            dropout = self.dropout,
            batch_first = False
        )

        self.sensor_encoder = torch.nn.TransformerEncoder(sensor_encoder_layers, self.nlayers)

        # self.mlp_sensor_encoder = torch.nn.Sequential(
        #     torch.nn.Linear(self.sensornet_out_size)
        # )

    def forward(self, x, src, static = None, src_key_padding_mask = None, attn_mask = None):
        '''
        x should be size (T, B, d)

        '''

        Ztime = self.time_encoder(x, 
            src_key_padding_mask = src_key_padding_mask,
            mask = attn_mask)

        # Move to (B, T, d):
        xIM = src.transpose(0, 1).unsqueeze(1)
        ZSN = self.sensor_conv(xIM).squeeze(1) # Will come out as (B, T, d)
        #print('ZSN', ZSN.permute(2, 0, 1).shape)
        Zsensor = self.sensor_encoder(ZSN.permute(2, 0, 1),
            src_key_padding_mask = None,
            mask = None)

        Zsensor = Zsensor.permute(2, 1, 0)#.max(dim=0).values
        #Zsensor = Zsensor.unsqueeze(0).repeat(Ztime.shape[0], 1, 1)

        # print('Ztime', Ztime.shape)
        #print('Zsensor', Zsensor.shape)

        Z = torch.cat([Ztime, Zsensor], dim = -1)

        return Z.transpose(0, 1) # flip to (B, T, d) before return


class DATEncPhi(torch.nn.Module):

    def __init__(self,
            d_inp,
            max_len,
            d_pe,
            MAX = 10000,
            nhead = 1,
            nlayers = 1,
            dim_feedforward = 72,
            dropout = 0.1,
            batch_first = False,
            cnn_kernel_len = 5,
            cnn_channels = 32,
        ):

        super(DATEncPhi, self).__init__()

        self.d_inp =d_inp
        self.max_len = max_len
        self.d_pe = d_pe

        self.encoder = DualAttentionTransformer(
            T = max_len,
            d = d_inp,
            d_pe = d_pe,
            nhead = nhead,
            nlayers = nlayers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = batch_first,
            cnn_kernel_len = cnn_kernel_len,
            cnn_channels = cnn_channels
        )

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

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
        output = self.encoder(x, src, src_key_padding_mask = overlen_mask)

        return output, overlen_mask