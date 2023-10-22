import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_MITECG

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

'''
Experiment 14 - alignment of label distributions
'''
pret_copy = False
pret_equal = False
print('Running variation pret_copy = {}, pret_equal = {}'.format(pret_copy, pret_equal))

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/mitecg_simple/models/transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

sim_criterion_label = LabelConsistencyLoss()
sim_criterion_cons = EmbedConsistencyLoss()

sim_criterion = [sim_criterion_cons, sim_criterion_label]

targs = transformer_default_args

for i in range(1, 6):
    trainEpi, val, test = process_MITECG(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Simple/')
    train_dataset = DatasetwInds(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 32
    targs['trans_dropout'] = 0.1
    targs['nlayers'] = 1
    targs['norm_embedding'] = False

    abl_params = AblationParameters(equal_g_gt = False,
        g_pret_equals_g = pret_copy, label_based_on_mask = True,
        ptype_assimilation = True, side_assimilation = True)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0
    }

    model = TimeXModel(
        d_inp = 1,
        max_len = val[0].shape[0],
        n_classes = 2,
        n_prototypes = 50,
        gsat_r = 0.5,
        transformer_args = targs,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict
    )

    model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    model.to(device)

    model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    # Initialize prototypes with training samples:
    model.init_prototypes(train = (trainEpi.X, trainEpi.time, trainEpi.y), stratify = True)

    if pret_copy:
        model.encoder_pret.load_state_dict(torch.load(tencoder_path.format(i)))

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/bc_w50p_stop_split={}.pt'.format(i, pret_copy, pret_equal)
    print('saving at', spath)

    #model = torch.compile(model)

    best_model = train_mv6_consistency(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        #exp_criterion = exp_criterion,
        sim_criterion = sim_criterion,
        beta_exp = 1.0,
        beta_sim = 1.0,
        val_tuple = val, 
        num_epochs = 100,
        save_path = spath,
        train_tuple = (trainEpi.X, trainEpi.time, trainEpi.y),
        early_stopping = False,
        selection_criterion = None,
        label_matching = True,
        embedding_matching = True
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()