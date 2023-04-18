import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv5 import train_mv5
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import Modelv5Univariate, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import SimCLRLoss
from txai.utils.predictors.select_models import cosine_sim, small_mask, sim_small_mask
from txai.utils.data.preprocess import process_Epilepsy

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

exp_criterion = [GSATLoss_Extended(r = 0.3)]

sim_criterion = SimCLRLoss()

targs = transformer_default_args

for i in range(1, 6):
    trainEpi, val, test = process_Epilepsy(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
    train_dataset = DatasetwInds(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

    train = (trainEpi.X, trainEpi.time, trainEpi.y)
    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 16
    targs['trans_dropout'] = 0.1
    targs['norm_embedding'] = False

    model = Modelv5Univariate(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 2,
        transformer_args = targs,
        trend_smoother = True,
    )
    model.encoder_main.load_state_dict(torch.load('models/transformer_split={}.pt'.format(i)))
    model.to(device)

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
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
        train_tuple = train,
        early_stopping = True,
        selection_criterion = cosine_sim,
        num_negatives = 32,
    )

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()