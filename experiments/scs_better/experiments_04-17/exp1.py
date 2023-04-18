import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6 import train_mv6
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import transformer_default_args
from txai.models.modelv6_v2 import Modelv6_v2, AblationParameters
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import SimCLRLoss, ConceptTopologyLoss, GeneralScoreContrastiveLoss
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

'''
Experiment 1 - Separate g vs. g_t
    - 3 variations: 
        1. g_t random, separate
        2. g = g_t (no separate training)
        3. g_t init w g, trained separate
'''
variation = 1
print('Running variation = {}'.format(variation))

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/models/Scomb_transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

#exp_criterion = [GSATLoss_Extended(r = 0.5)]
#exp_criterion = InterpretabilityCriterion(r = 0.5, lam = 1.0)

#exp_criterion = [SizeMaskLoss(mean = False, target_val = 5), PSizeLoss(max_len = 50)]
sim_criterion = SimCLRLoss()

targs = transformer_default_args

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 64
    targs['trans_dropout'] = 0.25
    targs['nlayers'] = 2
    targs['norm_embedding'] = False

    if variation == 1:
        abl_params = AblationParameters(equal_g_gt = False, trend_smoother_loss = False)
    elif variation == 2:
        abl_params = AblationParameters(equal_g_gt = True, trend_smoother_loss = False)
    elif variation == 3:
        abl_params = AblationParameters(equal_g_gt = False, trend_smoother_loss = False)

    model = Modelv6_v2(
        d_inp = 1,
        max_len = 200,
        n_classes = 4,
        n_concepts = 2,
        n_explanations = 2,
        gsat_r = 0.2,
        transformer_args = targs,
        trend_smoother = True,
        size_mask_target_val = 0.2,
        use_window = False,
        ablation_parameters = abl_params
    )

    model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    model.to(device)

    if variation == 3: # Init w main encoder weights
        model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    # for param in model.encoder_main.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/v6_exp1_var={}_split={}.pt'.format(variation,i)

    #model = torch.compile(model)

    best_model = train_mv6(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        #exp_criterion = exp_criterion,
        sim_criterion = sim_criterion,
        beta_exp = 1.0,
        beta_sim = 1.0,
        val_tuple = val, 
        num_epochs = 50,
        save_path = spath,
        train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        early_stopping = False,
        selection_criterion = None,
        num_negatives = 64,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()