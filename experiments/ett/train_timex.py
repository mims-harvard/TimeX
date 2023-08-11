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
    timex_model = TimeXForecasting(
        transformer_config = transformer_kwargs,
        masktoken_stats = masktoken_stats,
        pooling_method = 'max',
        r = 0.5
    )

    # Initialize timex with encoder:
    timex_model.encoder_main.load_state_dict(predictor.state_dict())
    for p in timex_model.encoder_main.parameters():
        p.requires_grad = False

    timex_model.encoder_t.load_state_dict(predictor.state_dict())

    timex_model.to(device)

    mbc_loss_fn = EmbedConsistencyLoss()
    lc_loss_fn = LabelConsistencyLoss_Forecasting()

    optim = torch.optim.AdamW(timex_model.parameters(), lr = 1e-4, weight_decay = 0.001)

    # train
    print("training")
    for iteration in trange(epochs):
        losses = []
        for (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in train_dl:
            batch = batch_x.float().to(device)
            target = batch_y.float().to(device)

            # NOTE shapes:
            # batch (input) = [batch size, input sequence len = 720, features=1]
            # target        = [batch size, target sequence len = 24, features=1]
            # embedding (if get_embedding=True in model()) = [batch_size, input sequence len = 720, dim = 32]
            out = timex_model(batch)

            # Call model loss:
            exp_loss = timex_model.compute_loss(out)
            mbc_loss = mbc_loss_fn(out["all_z"][0], out["all_z"][1])
            lc_loss = lc_loss_fn(out['pred_mask'], out['pred'])

            loss = exp_loss + mbc_loss + lc_loss

            torch.nn.utils.clip_grad_norm_(timex_model.parameters(), 1.0)

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        
    timex_model.save_state(path = 'timex.pt')

if __name__ == "__main__":
    main()