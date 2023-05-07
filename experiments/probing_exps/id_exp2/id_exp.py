import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import transformer_default_args
from txai.models.modelv6_v2 import AblationParameters
from txai.models.eliminates.modelv6_v2_for_idexp import Modelv6_v2
from txai.trainers.eliminates.train_mv6_consistency_idexp import train_mv6_consistency_idexp
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4_idexp
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

'''
Experiment 14 - alignment of label distributions
'''
pret_copy = False
pret_equal = False
print('Running variation pret_copy = {}, pret_equal = {}'.format(pret_copy, pret_equal))

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/models/Scomb_transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

def make_external_identifier(train, val, test):
    train_X = train[0].clone()
    train_times = train[1].clone()
    train_y = train[2].clone()
    train_inds = torch.arange(train_X.shape[1])

    val_X = val[0].clone()
    val_times = val[1].clone()
    val_y = val[2].clone()
    val_inds = torch.arange(val_X.shape[1]) + 1 + train_inds.max()

    test_X = test[0].clone()
    test_times = test[1].clone()
    test_y = test[2].clone()
    test_inds = torch.arange(test_X.shape[1]) + 1 + val_inds.max()

    whole_X = torch.cat([train_X, val_X, test_X], dim = 1)
    
    # Find identifiers for the whole X pattern:
    comb_inds = torch.combinations(torch.arange(whole_X.shape[0]), r = 2, with_replacement = False)

    for i in range(whole_X.shape[1]):
        j, k = comb_inds[i,0], comb_inds[i,1]
        whole_X[j, i, 0] = 5.0
        whole_X[k, i, 0] = 5.0

    # Sample into inds:
    external_train = (whole_X[:,train_inds,:], train_times, train_y)
    external_val = (whole_X[:,val_inds,:], val_times, val_y)
    external_test = (whole_X[:,test_inds,:], test_times, test_y)

    return external_train, external_val, external_test

#exp_criterion = [GSATLoss_Extended(r = 0.5)]
#exp_criterion = InterpretabilityCriterion(r = 0.5, lam = 1.0)

#exp_criterion = [SizeMaskLoss(mean = False, target_val = 5), PSizeLoss(max_len = 50)]
#sim_criterion = LabelSimLoss()
sim_criterion = ConceptConsistencyLoss()

targs = transformer_default_args

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    external_train, external_val, external_test = make_external_identifier((D['train_loader'].X, D['train_loader'].times, D['train_loader'].y), val, test)

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 64
    targs['trans_dropout'] = 0.25
    targs['nlayers'] = 2
    targs['norm_embedding'] = False


    abl_params = AblationParameters(equal_g_gt = False, trend_smoother_loss = False,
        g_pret_equals_g = pret_equal)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0
    }

    model = Modelv6_v2(
        d_inp = 1,
        max_len = 200,
        n_classes = 4,
        n_concepts = 2,
        n_explanations = 2,
        gsat_r = 0.2,
        tau = 2.0,
        transformer_args = targs,
        trend_smoother = False,
        size_mask_target_val = 0.2,
        use_window = False,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict
    )

    model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    model.to(device)

    model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    if pret_copy:
        model.encoder_pret.load_state_dict(torch.load(tencoder_path.format(i)))

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/ours_split={}_noequal.pt'.format(i)

    #model = torch.compile(model)

    best_model = train_mv6_consistency_idexp(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        #exp_criterion = exp_criterion,
        sim_criterion = sim_criterion,
        beta_exp = 2.0,
        beta_sim = 1.0,
        val_tuple = val, 
        num_epochs = 200,
        save_path = spath,
        train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        early_stopping = False,
        selection_criterion = None,
        label_matching = False,
        external_train_X = external_train[0],
        external_val_tuple = external_val
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4_idexp(test, external_test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()