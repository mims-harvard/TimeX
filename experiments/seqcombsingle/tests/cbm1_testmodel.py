import torch
import numpy as np

from txai.models.cbmv1 import CBMv1
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def gen_dataset(samps = 1000):
    inc = torch.cat([gen_sample(X, increase = True) for _ in range(samps)], dim = 1).to(device)
    dec = torch.cat([gen_sample(X, increase = False) for _ in range(samps)], dim = 1).to(device)
    
    times = torch.arange(inc.shape[0]).unsqueeze(-1).repeat(1, samps * 2).to(device)
    whole = torch.cat([inc, dec], dim=1)
    batch_id = torch.cat([torch.zeros(inc.shape[1]), torch.ones(dec.shape[1])]).to(device).long()
    return whole, times, batch_id

def test_one_forward():

    # Generate shape concept bank:
    samps = 1000
    X = torch.randn(50, 1, 1).to(device)
    inc = torch.cat([gen_sample(X, increase = True) for _ in range(samps)], dim = 1).to(device)
    dec = torch.cat([gen_sample(X, increase = False) for _ in range(samps)], dim = 1).to(device)
    times = torch.arange(inc.shape[0]).unsqueeze(-1).repeat(1, samps * 2).to(device)

    D = process_Synth(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    model = CBMv1(
        d_inp = 1,
        max_len = 50,
        n_classes = 4,
        n_concepts = 2,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        norm_embedding = False,
        type_masktoken = 'norm_datawide',
        masktoken_kwargs = {'mu': mu, 'std': std},
        mask_seed = 1234,
    )
    model.to(device)

    model.encoder.load_state_dict(torch.load('models/Scomb_transformer_split=1.pt'))

    print('inc', inc.shape)
    print('dec', dec.shape)
    batch_id = torch.cat([torch.zeros(inc.shape[1]), torch.ones(dec.shape[1])]).to(device).long()
    print('batch id', batch_id.shape)
    model.store_concept_bank(torch.cat([inc, dec], dim=1), times, batch_id = batch_id)

    # Get dummy:
    times = torch.arange(50).unsqueeze(-1).repeat(1, 64).to(device)
    X = torch.randn(50, 64, 1).to(device)
    print('Times', times.shape)
    print('X', X.shape)

    yhat, concept_scores, mask, logits = model(X, times)

    print('Output shapes ------------------')
    print('yhat         ', yhat.shape)
    print('concept score', concept_scores.shape)
    print('concept score', concept_scores[0,:])
    print('mask[0]      ', mask[0].shape)
    print('logits[0]    ', logits[0].shape)

if __name__ == '__main__':
    main()