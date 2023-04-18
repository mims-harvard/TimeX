import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv5 import train_mv5
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import Modelv5Univariate, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import SimCLRLoss
from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

exp_criterion = [GSATLoss_Extended(r = 0.5)]
#exp_criterion = InterpretabilityCriterion(r = 0.5, lam = 1.0)

sim_criterion = SimCLRLoss()

targs = transformer_default_args

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 16
    targs['trans_dropout'] = 0.1
    targs['norm_embedding'] = False

    model = Modelv5Univariate(
        d_inp = 1,
        max_len = 50,
        n_classes = 4,
        transformer_args = targs,
        trend_smoother = True,
    )
    model.encoder_main.load_state_dict(torch.load('models/Scomb_transformer_split={}.pt'.format(i)))
    model.to(device)

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/v5_smooth_split={}.pt'.format(i)

    best_model = train_mv5(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        exp_criterion = exp_criterion,
        sim_criterion = sim_criterion,
        beta_exp = torch.tensor([2.0]),
        beta_sim = 1.0,
        val_tuple = val, 
        num_epochs = 300,
        save_path = spath,
        train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        early_stopping = True,
        selection_criterion = cosine_sim,
        num_negatives = 64,
        #loss_uses_dict = True,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()