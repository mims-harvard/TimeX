import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv3_embedsim import train_mv3_embedsim
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv3 import Modelv3EmbSim
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv3_sim
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_Epilepsy

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

exp_criterion = [SizeMaskLoss()]


for i in range(1, 6):
    trainEpi, val, test = process_Epilepsy(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
    train_dataset = EpiDataset(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

    model = Modelv3EmbSim(
        d_inp = 1,
        max_len = val.X.shape[0],
        n_classes = 2,
        trans_dim_feedforward = 64,
        trans_dropout = 0.1,
        norm_embedding = False,
        trend_smoother = False,
    )
    #model.encoder_main.load_state_dict(torch.load('models/Scomb_transformer_split={}.pt'.format(i)))
    model.to(device)

    # for param in model.encoder_main.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/v3sep_split={}.pt'.format(i)

    best_model = train_mv3_embedsim(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        exp_criterion = exp_criterion,
        sim_criterion = torch.nn.CosineSimilarity(dim=-1),
        beta_exp = torch.tensor([0.5]),
        beta_sim = 4.0,
        val_tuple = (val.X, val.time, val.y), 
        num_epochs = 300,
        save_path = spath,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv2_sim(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()