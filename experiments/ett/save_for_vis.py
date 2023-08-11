import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange

from model import Transformer
from timex_forecasting import TimeXForecasting
from dataset import Dataset_ETT_hour

from txai.utils.predictors.loss_cl import EmbedConsistencyLoss, LabelConsistencyLoss_Forecasting

def main():

    torch.manual_seed(0)

    # hyperparams selected from univariate transformer:
    # https://github.com/moraieu/query-selector/blob/master/settings/tuned/ts_full_u_h1_24.json
    seq_len = 720
    pred_len = 24
    batch_size = 64
    epochs = 50

    transformer_kwargs = {
        "dim_val": 32,
        "dim_attn": 128,
        "input_size": 1,
        "dec_seq_len": 48,
        "out_seq_len": pred_len,
        "output_len": 1,
        "n_decoder_layers": 3,
        "n_heads": 4,
        "n_encoder_layers": 3,
        "enc_attn_type": "full",
        "dec_attn_type": "full",
        "dropout": 0.1,
    }

    predictor = Transformer(**transformer_kwargs)

    train_ds = Dataset_ETT_hour(root_path="./data", data_path="ETTh1.csv", flag="train", size=[seq_len, 0, pred_len], features="S")
    test_ds = Dataset_ETT_hour(root_path="./data", data_path="ETTh1.csv", flag="test", size=[seq_len, 0, pred_len], features="S")

    X = torch.stack([torch.from_numpy(train_ds[i][0]) for i in range(len(train_ds))], dim = 0) # Access all elements
    mu = X.mean(dim=0)
    std = X.std(unbiased=True,dim=0)
    masktoken_stats = (mu, std)

    print("mu shape", mu.shape)
    print("std shape", std.shape)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictor.load_state_dict(torch.load("predictor.pt"))
    predictor.to(device)

    #optim = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-2)

    # Setup TimeX:
    sd, config = torch.load('timex.pt')

    timex_model = TimeXForecasting(**config)

    timex_model.load_state_dict(sd)

    timex_model.to(device)

    # Inference on testing samples:
    Xall, exp_all = [], []
    preds, trues = [], []
    with torch.no_grad():
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_dl:
            batch = batch_x.float().to(device)
            target = batch_y.float().to(device)
            out = timex_model(batch)

            result = out['pred']

            Xall.append(batch.detach().clone().cpu())
            exp_all.append(out['mask_logits'].detach().clone().cpu())
            preds.append(result.detach().clone().cpu())
            trues.append(target.cpu())

    Xfull = torch.cat(Xall, dim = 0).cpu()
    expfull = torch.cat(exp_all, dim = 0).cpu()
    predsfull = torch.cat(preds, dim = 0).cpu()
    truesfull = torch.cat(trues, dim = 0).cpu()

    torch.save((Xfull, expfull, predsfull, truesfull), 'timex_output.pt')
    

if __name__ == '__main__':
    main()