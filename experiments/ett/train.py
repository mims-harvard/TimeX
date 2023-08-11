import torch
from torch.utils.data import DataLoader
import numpy as np

from model import Transformer
from dataset import Dataset_ETT_hour

def main():

    torch.manual_seed(0)

    # hyperparams selected from univariate transformer:
    # https://github.com/moraieu/query-selector/blob/master/settings/tuned/ts_full_u_h1_24.json
    seq_len = 720
    pred_len = 24
    batch_size = 64
    iterations = 5

    model = Transformer(
        dim_val=32,
        dim_attn=128,
        input_size=1,
        dec_seq_len=48,
        out_seq_len=pred_len,
        output_len=1,
        n_decoder_layers=3,
        n_heads=4,
        n_encoder_layers=3,
        enc_attn_type="full",
        dec_attn_type="full",
        dropout=0.1,
    )

    train_ds = Dataset_ETT_hour(root_path="./data", data_path="ETTh1.csv", flag="train", size=[seq_len, 0, pred_len], features="S")
    test_ds = Dataset_ETT_hour(root_path="./data", data_path="ETTh1.csv", flag="test", size=[seq_len, 0, pred_len], features="S")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-2)


    # train
    print("training")
    for iteration in range(iterations):
        losses = []
        for (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in train_dl:
            batch = batch_x.float().to(device)
            target = batch_y.float().to(device)

            # NOTE shapes:
            # batch (input) = [batch size, input sequence len = 720, features=1]
            # target        = [batch size, target sequence len = 24, features=1]
            # embedding (if get_embedding=True in model()) = [batch_size, input sequence len = 720, dim = 32]
            result = model(batch, get_embedding=False)

            loss = torch.nn.functional.mse_loss(result.squeeze(2), target.squeeze(2))
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"iteration: {iteration}, loss: {torch.tensor(losses).mean():.3f}")
        

    # test
    print("testing")
    preds, trues = [], []
    with torch.no_grad():
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_dl:
            batch = batch_x.float().to(device)
            target = batch_y.float().to(device)
            result = model(batch)
            preds.append(result.cpu().numpy())
            trues.append(target.cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")

    torch.save(model.state_dict(), "predictor.pt")

if __name__ == "__main__":
    main()