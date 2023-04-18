import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv3 import train_mv3
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv3 import Modelv3SepExtractor
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv3
from txai.synth_data.simple_spike import SpikeTrainDataset

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

exp_criterion = [SizeMaskLoss()]


for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    model = Modelv3SepExtractor(
        d_inp = 1,
        max_len = 50,
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        norm_embedding = False,
        trend_smoother = False,
    )
    model.encoder_main.load_state_dict(torch.load('models/Scomb_transformer_split={}.pt'.format(i)))
    model.to(device)

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/Scomb_v3sep_split={}.pt'.format(i)

    best_model = train_mv3(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        exp_criterion = exp_criterion,
        logit_criterion = torch.nn.KLDivLoss(reduction='batchmean'),
        beta_exp = torch.tensor([2.0]),
        beta_logit = 4.0,
        val_tuple = val, 
        num_epochs = 300,
        save_path = spath,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv2(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()