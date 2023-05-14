import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_Boiler_OLD, RWDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,#torch.tensor([1.0,4.0]),
    reduction = 'mean'
)

for i in range(1, 6):
    torch.cuda.empty_cache()
    trainB, val, test = process_Boiler_OLD(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Boiler/')
    train_dataset = RWDataset(*trainB)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 2,
        nlayers = 1,
        trans_dim_feedforward = 32,
        trans_dropout = 0.25,
        d_pe = 16,
        norm_embedding = True,
        stronger_clf_head = False,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/transformer_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 2,
        num_epochs = 500,
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