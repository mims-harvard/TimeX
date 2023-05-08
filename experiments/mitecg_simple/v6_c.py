import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.modelv5 import transformer_default_args
#from txai.models.modelv6_v2 import AblationParameters
from txai.models.modelv6_v2_concepts import Modelv6_v2_concepts, AblationParametersConcepts
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.concepts import *
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_MITECG
#from txai.utils.predictors.select_models import cosine_sim

from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero

'''
Experiment 3 - concept distribution changes
'''
n_aug = 50
gauss_std = 1.0
rand_mask = 0.4
mshift = False
print('Running variation std={} rm={} ms={}'.format(gauss_std, rand_mask, mshift))

tencoder_path = "/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/mitecg_simple/models/transformer_split={}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

# Load concept tuples:
def get_concepts():
    chosen_concepts = torch.load('/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/mitecg_simple/golden_concepts_tuple.pt')
    Xcon, Tcon, Mcon = chosen_concepts

    gblur = GaussianBlurParams(std = gauss_std)
    #tshift = TimeShiftParams(p = 0.5, max_abs_shift = 10)
    tshift = None
    if mshift:
        mask_shift = MaskShiftParams(p = 0.5, max_abs_shift = 10)
    else:
        mask_shift = None
    rmask = RandomMaskParams(p = rand_mask)

    C = ConceptsWithAugmentations(
        X = Xcon, 
        times = Tcon,
        masks = Mcon,
        gaussian_blur_params = gblur,
        time_shift_params = tshift,
        mask_shift_params = mask_shift,
        random_mask_params = rmask,
    )

    C.to(device)

    return C


#chosen_concepts.to(device)


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

    # Calc statistics for baseline:
    # mu = D['train_loader'].X.mean(dim=1)
    # std = D['train_loader'].X.std(unbiased = True, dim = 1)

    # Change transformer args:
    targs['trans_dim_feedforward'] = 32
    targs['trans_dropout'] = 0.1
    targs['nlayers'] = 1
    targs['norm_embedding'] = False


    abl_params = AblationParametersConcepts(equal_g_gt = False, g_pret_equals_g = False, label_based_on_mask = True,
        ze_closeness_to_concepts = True, concept_matching = True)

    loss_weight_dict = {
        'gsat': 1.0,
        'connect': 2.0,
        'entropy': 0.0,
    }

    model = Modelv6_v2_concepts(
        d_inp = 1,
        max_len = val[0].shape[0],
        n_classes = 2,
        n_concepts = 5,
        n_explanations = 1,
        gsat_r = 0.5,
        transformer_args = targs,
        size_mask_target_val = 0.2,
        use_window = False,
        ablation_parameters = abl_params,
        loss_weight_dict = loss_weight_dict,
        predefined_concepts = get_concepts(),
        n_aug_per_concept = n_aug,
    )

    model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
    model.to(device)

    model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

    # if pret_copy:
    #     model.encoder_pret.load_state_dict(torch.load(tencoder_path.format(i)))

    for param in model.encoder_main.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
    
    spath = 'models/v6_std={}_rm={}_ms={}_split={}_no_concept.pt'.format(gauss_std, rand_mask, mshift, i)
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

    # Get concept embeddings and save:
    #z_concept = model.get_concept_embeddings(n_aug_per_concept = 100)
    #torch.save(z_concept.detach().clone().cpu(), 'embeddings/model2_embeds.pt')

    exit()