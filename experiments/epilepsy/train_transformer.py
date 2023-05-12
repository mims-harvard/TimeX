import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_Epilepsy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

for i in range(1, 6):
    torch.cuda.empty_cache()
    trainEpi, val, test = process_Epilepsy(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
    train_dataset = EpiDataset(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
    
    print('X shape')
    print(trainEpi.X.shape)
    print('y shape', trainEpi.y.shape)

    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 2,
        nlayers = 1,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
        norm_embedding = False,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/transformer_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 2,
        num_epochs = 300,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        validate_by_step = 32,
        use_scheduler = False
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, 'models/transformer_split={}_cpu.pt'.format(i))

    f1 = eval_mvts_transformer(test, model, batch_size = 32)
    print('Test F1: {:.4f}'.format(f1))