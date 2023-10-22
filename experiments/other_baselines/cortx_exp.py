import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#from txai.trainers.train_mv6_trip import train_mv6_trip
from txai.trainers.train_mv6_consistency import train_mv6_consistency
from txai.utils.evaluation import ground_truth_xai_eval
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.shapebank.v1 import gen_dataset, gen_dataset_zero
from contrastive_model import DualBranchContrast
from contrast_generator import contrast_generator
from txai.models.mask_generators.maskgen import MaskGenerator
from txai.models.modelv2 import MaskGenStochasticDecoder_NoCycleParam
import numpy as np
import infonce as L
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pret_copy = False
pret_equal = True
print('Running variation pret_copy = {}, pret_equal = {}'.format(pret_copy, pret_equal))

tencoder_path = "/home/huan/Documents/TimeSeriesCBM-main/experiments/freqshape/models/Scomb_transformer_split={}.pt"

device = 'cuda'


def one_hot_ce_loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    _, labels = torch.max(targets, dim=1)
    return criterion(outputs, labels)

class BoilerDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y

for i in [4]:
    D = process_Synth(split_no = i, device = device, base_path = '/home/huan/Documents/TimeSeriesCBM-main/FreqShape')
    dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    predict_model =  TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    ).to(device)

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    ).to(device)

    predict_model.load_state_dict(torch.load(tencoder_path.format(i)))
    model.load_state_dict(torch.load(tencoder_path.format(i)))
    model.mlp.requires_grad_ = False
    contrast_gen = contrast_generator(predict_model)
    ccontras_loss = DualBranchContrast(loss=L.InfoNCE(tau=0.7), mode='G2G')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    best_l2 = 1.0
    best_ste_l2 = 0.0
    model.train()
    pos_num = 100
    neg_num = 100
    for epoch in range(100):
        train_loss = 0.0 
        for X, times, y in train_loader:
            optimizer.zero_grad()
            tar, pos = contrast_gen(model, X, pos_num, neg_num, times)
            loss = ccontras_loss(g1=pos, g2=tar.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_loader.dataset)
        print(train_loss)

    # model_sdict_cpu = {k:v for k, v in  model.state_dict().items()}
    # torch.save(model_sdict_cpu, 'models/transformer_split={}.pt'.format(i))

    d_inp = 1
    d_pe = 16
    max_len = val[0].shape[0]
    decoder = MaskGenerator(d_z = (d_inp + d_pe), max_len = max_len, \
                            tau = 1,  use_ste = False).to('cuda')
    # decoder = nn.ModuleList()
    # for _ in range(2):
    #     mgen = MaskGenStochasticDecoder_NoCycleParam(d_z = (d_inp + d_pe), max_len = max_len, tau = 1, 
    #         use_ste = False)
    #     decoder.append(mgen)
    cri = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    for epoch in range(50):
        train_loss = 0.0 
        for X, times, y in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                z_seq = model.embed(X, times, captum_input = False, aggregate = False)
            mask_in, ste_mask = decoder(z_seq, X, times)
            loss = cri(mask_in.permute(1,0,2), X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_loader.dataset)
        print(train_loss)



    z_seq = model.embed(train_loader.dataset.X, train_loader.dataset.times, captum_input = False, aggregate = False)

    mask_in, ste_mask = decoder(z_seq, train_loader.dataset.X, train_loader.dataset.times)



    y = test[2]
    
    X = test[0][:,(y != 0),:]
    times = test[1][:,y != 0]
    gt_exps = D['gt_exps'][:,(y != 0).detach().cpu(),:]
    z_seq = model.embed(X, times, captum_input = False, aggregate = False)

    mask_in, ste_mask = decoder(z_seq, X, times)
    results_dict = ground_truth_xai_eval(mask_in.permute(1,0,2), gt_exps)


    for k, v in results_dict.items():
        print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))
