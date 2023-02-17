import torch
import numpy as np
import matplotlib.pyplot as plt

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
#from txai.utils.predictors.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.shapebank.v1 import gen_sample_zero

from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerMVTS(
    d_inp = 1,
    max_len = 50,
    n_classes = 4,
    trans_dim_feedforward = 16,
    trans_dropout = 0.1,
)

model.load_state_dict(torch.load('seqcombsingle/models/Scomb_transformer_split=1.pt'))
model.to(device)

X = torch.randn(50, 1, 1).to(device)
time = torch.arange(50).unsqueeze(-1).to(device)

print(model.embed(X, time).shape)

# How to generate shape bank?
def gen_sample(template, increase = True):

    length = np.random.choice(np.arange(start=5, stop=45))
    if increase == True:
        seq = np.linspace(-2, 2, num = int(length))
    else:
        seq = np.linspace(2, -2, num = int(length))

    seq *= np.random.normal(1.0, scale = 0.01, size = seq.shape)

    # Get mask w/sampled location:
    loc = np.random.choice(np.arange(start=0, stop=int(template.shape[0]-length)))

    a = torch.randn_like(template)
    a[loc:(loc+length),0,0] = torch.from_numpy(seq)

    return a


increase_sample_bank = [gen_sample_zero(X) for _ in range(1000)]
increases = torch.cat(increase_sample_bank, dim=1)
time_rep = time.repeat(1, increases.shape[1])

increase_embed = model.embed(increases, time_rep)

decrease_sample_bank = [gen_sample_zero(X, increase = False) for _ in range(1000)]
decreases = torch.cat(decrease_sample_bank, dim=1)
time_rep = time.repeat(1, decreases.shape[1])

decrease_embed = model.embed(decreases, time_rep)

embed = torch.cat([increase_embed, decrease_embed], dim=0)

tsne_inc = TSNE().fit_transform(embed.detach().cpu().numpy())

plt.scatter(tsne_inc[:1000,0], tsne_inc[:1000,1], c = 'orange', label = 'Increasing')
plt.scatter(tsne_inc[1000:,0], tsne_inc[1000:,1], c = 'green', label = 'Decreasing')
plt.title('Increasing vs. Decreasing Masked Sequences')
plt.legend()
#plt.savefig('inc_v_dec.png', dpi=200)
plt.show()





