import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency
from txai.trainers.train_mv6_ptype import train_mv6_ptype

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import transformer_default_args
from txai.models.modelv6_v3 import Modelv6_v3
from txai.models.modelv6_v2_ptype import Modelv6_v2_ptype, AblationParametersPrototype
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import SimCLRLoss, ConceptTopologyLoss, GeneralScoreContrastiveLoss, EmbedConsistencyLoss
from txai.utils.concepts import PreChosenConceptList 
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better/models/Scomb_transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

targs = transformer_default_args

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    # Change transformer args:
    targs['trans_dim_feedforward'] = 128
    targs['trans_dropout'] = 0.1
    targs['nlayers'] = 2
    targs['norm_embedding'] = False

    abl_params = AblationParametersPrototype(equal_g_gt = True, trend_smoother_loss = False, concept_matching = True,
        hard_concept_matching = False, use_loss_on_concept_sims = False,
        use_concept_corr_loss = False)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0
    }

    model = Modelv6_v2_ptype(
        d_inp = 1,
        max_len = 200,
        n_classes = 4,
        n_explanations = 2,
        n_prototypes = 2,
        gsat_r = 0.5,
        transformer_args = targs,
        trend_smoother = False,
        size_mask_target_val = 0.2,
        use_window = False,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict,
    )
    model.to(device)

    # model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    # model.to(device)

    # model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    # for param in model.encoder_main.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/trial_split={}.pt'.format(i)

    #model = torch.compile(model)

    best_model = train_mv6_ptype(
        model,
        optimizer = optimizer,
        train_loader = train_loader,
        clf_criterion = clf_criterion,
        beta_exp = 0.5,
        val_tuple = val, 
        num_epochs = 200,
        save_path = spath,
        train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        early_stopping = False,
        selection_criterion = None,
    )

    #print(model.distance_mlp.get_parameter('0.weight'))

    sdict, config = torch.load(spath)

    model.load_state_dict(sdict)

    f1, _ = eval_mv4(test, model)
    print('Test F1: {:.4f}'.format(f1))
    exit()