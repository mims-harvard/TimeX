import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from txai.trainers.train_mv6_consistency import train_mv6_consistency
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import transformer_default_args
from txai.models.modelv6_v2 import Modelv6_v2, AblationParameters
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import SimCLRLoss, ConceptTopologyLoss, GeneralScoreContrastiveLoss, EmbedConsistencyLoss
from txai.utils.data.preprocess import process_Epilepsy
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

#exp_criterion = [GSATLoss_Extended(r = 0.5)]
#exp_criterion = InterpretabilityCriterion(r = 0.5, lam = 1.0)

#exp_criterion = [SizeMaskLoss(mean = False, target_val = 5), PSizeLoss(max_len = 50)]
sim_criterion = EmbedConsistencyLoss()

targs = transformer_default_args

for i in range(1, 6):
    trainEpi, val, test = process_Epilepsy(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
    train_dataset = DatasetwInds(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

    train = (trainEpi.X, trainEpi.time, trainEpi.y)
    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    # Calc statistics for baseline:
    # mu = D['train_loader'].X.mean(dim=1)
    # std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 16
    targs['trans_dropout'] = 0.1
    targs['norm_embedding'] = False

    abl_params = AblationParameters(equal_g_gt = False, trend_smoother_loss = False, concept_matching = True,
        use_loss_on_concept_sims = True, use_concept_corr_loss = True, hard_concept_matching = False)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0
    }

    model = Modelv6_v2(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 2,
        n_concepts = 2,
        n_explanations = 2,
        gsat_r = 0.5,
        transformer_args = targs,
        trend_smoother = False,
        use_window = False,
        size_mask_target_val = 0.2,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict
    )

    model.encoder_main.load_state_dict(torch.load('models/transformer_split={}.pt'.format(i)))
    model.to(device)

    model.encoder_t.load_state_dict(torch.load('models/transformer_split={}.pt'.format(i)))

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
    
    spath = 'models/v6_concepts_split={}.pt'.format(i)

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
        train_tuple = train,
        early_stopping = False,
        selection_criterion = None,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()