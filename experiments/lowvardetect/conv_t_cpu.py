import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for i in range(1, 6):
    #D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
    #train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    #val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = 2,
        max_len = 200,
        n_classes = 4,
        nlayers = 1,
        trans_dim_feedforward = 32,
        trans_dropout = 0.25,
        d_pe = 16,
        # aggreg = 'mean',
        norm_embedding = True
    )
    
    spath = 'models/transformer_new2_split={}.pt'.format(i)
    print('re-save {}'.format(spath))
    sd = torch.load(spath)

    model_sdict_cpu = {k:v.cpu() for k, v in  sd.items()}
    torch.save(model_sdict_cpu, 'models/transformer_split={}_cpu.pt'.format(i))

