import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.bc_model import BCExplainModel, AblationParameters, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pret_copy = False
pret_equal = False
print('Running variation pret_copy = {}, pret_equal = {}'.format(pret_copy, pret_equal))

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/seqcomb_mv/models/transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

sim_criterion_label = LabelConsistencyLoss()
sim_criterion_cons = EmbedConsistencyLoss()

sim_criterion = [sim_criterion_cons, sim_criterion_label]

targs = transformer_default_args

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombMV')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 128
    targs['trans_dropout'] = 0.25
    targs['nlayers'] = 2
    targs['norm_embedding'] = False

    abl_params = AblationParameters(equal_g_gt = False,
        g_pret_equals_g = pret_copy, label_based_on_mask = True)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0
    }

    model = BCExplainModel(
        d_inp = 4,
        max_len = 200,
        n_classes = 4,
        n_prototypes = 2,
        gsat_r = 0.5,
        transformer_args = targs,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict,
        masktoken_stats = (mu, std)
    )

    model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    model.to(device)

    model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    if pret_copy:
        model.encoder_pret.load_state_dict(torch.load(tencoder_path.format(i)))

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/bc_stronger_split={}.pt'.format(i, pret_copy, pret_equal)
    print('saving at', spath)

    #model = torch.compile(model)

    best_model = train_mv6_consistency(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        #exp_criterion = exp_criterion,
        sim_criterion = sim_criterion,
        beta_exp = 2.0,
        beta_sim = 1.0,
        val_tuple = val, 
        num_epochs = 100,
        save_path = spath,
        train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
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