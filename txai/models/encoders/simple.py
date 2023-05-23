from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(d_inp, out_channels=dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if captum_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            # batch, time, channels -> batch, channels, time
            x = x.permute(0, 2, 1)
        else: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, channels, time
            x = x.permute(1, 2, 0)

        if x.shape[-1] < 8:
            # pad sequence to at least 8 so two max pools don't fail
            # necessary for when WinIT uses a small window
            x = F.pad(x, (0, 8 - x.shape[-1]), mode="constant", value=0)

        embedding = self.encoder(x)
        out = self.mlp(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out


class LSTM(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.encoder = nn.LSTM(
            d_inp,
            dim // 2, # half for bidirectional
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if not captum_input: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, time, channels
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        embedding, _ = self.encoder(x)
        embedding = embedding.mean(dim=1) # mean over time
        out = self.mlp(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out
