import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_MITECG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,#torch.tensor([1.0, 3.0]),
    reduction = 'mean'
)

for i in range(1, 6):
    torch.cuda.empty_cache()
    trainEpi, val, test, _ = process_MITECG(split_no = i, device = device, hard_split = True, normalize = True, balance_classes = True,
        base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Hard/')
    train_dataset = EpiDataset(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
    
    print('X shape')
    print(trainEpi.X.shape)
    print('y shape', trainEpi.y.shape)

    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    # Look at y's:
    # print((val[-1] == 0).sum())
    # print((val[-1] == 1).sum())
    # print((trainEpi.y == 0).sum())
    # print((trainEpi.y == 1).sum())
    # exit()

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 2,
        nlayers = 2,
        nhead = 1,
        trans_dim_feedforward = 32,
        trans_dropout = 0.1,
        #enc_dropout = 0.1,
        d_pe = 16,
        stronger_clf_head = True,
        pre_agg_transform = False,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    print('lr', 5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/transformer_trial_split={}.pt'.format(i)
    print('Saving at {}'.format(spath))

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 2,
        num_epochs = 1000,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        validate_by_step = 32,
        use_scheduler = False,
        print_freq = 1
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, 'models/transformer_split={}_cpu.pt'.format(i))

    f1 = eval_mvts_transformer(test, model, batch_size = 32)
    print('Test F1: {:.4f}'.format(f1))