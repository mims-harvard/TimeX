import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X
        self.times = times
        self.y = y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

def make_external_identifier(train, val, test):
    train_X = train[0].clone()
    train_times = train[1].clone()
    train_y = train[2].clone()
    train_inds = torch.arange(train_X.shape[1])

    val_X = val[0].clone()
    val_times = val[1].clone()
    val_y = val[2].clone()
    val_inds = torch.arange(val_X.shape[1]) + 1 + train_inds.max()

    test_X = test[0].clone()
    test_times = test[1].clone()
    test_y = test[2].clone()
    test_inds = torch.arange(test_X.shape[1]) + 1 + val_inds.max()

    whole_X = torch.cat([train_X, val_X, test_X], dim = 1)
    
    # Find identifiers for the whole X pattern:
    comb_inds = torch.combinations(torch.arange(whole_X.shape[0]), r = 2, with_replacement = False)

    for i in range(whole_X.shape[1]):
        j, k = comb_inds[i,0], comb_inds[i,1]
        whole_X[j, i, 0] = 5.0
        whole_X[k, i, 0] = 5.0

    # Sample into inds:
    external_train = (whole_X[:,train_inds,:], train_times, train_y)
    external_val = (whole_X[:,val_inds,:], val_times, val_y)
    external_test = (whole_X[:,test_inds,:], test_times, test_y)

    return external_train, external_val, external_test

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')

    val, test = D['val'], D['test']

    train_data, val, test = make_external_identifier((D['train_loader'].X, D['train_loader'].times, D['train_loader'].y), val, test)

    train_loader = torch.utils.data.DataLoader(SimpleDataset(*train_data), batch_size = 64, shuffle = True)

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        nlayers = 2,
        n_classes = 4,
        enc_dropout = 0.5,
        trans_dim_feedforward = 64,
        trans_dropout = 0.5,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.01)
    
    spath = 'models/transformer_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 4,
        num_epochs = 200,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, 'models/transformer_split={}_cpu.pt'.format(i))

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))

    if i > 2:
        break