import math
from copy import deepcopy
import pickle

import torch
import torch.nn.functional as F
from fast_transformers.builders import TransformerEncoderBuilder
from performer_pytorch import Performer
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)


class CogPilotData(torch.utils.data.Dataset):
    def __init__(self, data_path, split="train"):
        super().__init__()
        assert split in ["train", "val", "test"]
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # NOTE: data is stored [sequence len, N, features]
        self.x = torch.from_numpy(data[f"P{split}"]).float()
        self.y = torch.from_numpy(data[f"y{split}"]).long()
        self.s = torch.from_numpy(data[f"s{split}"]).long()
        self.variables = [b for a in data["Vars"] for b in a]  # flatten

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[:, idx], self.y[idx], self.s[idx]


class Model(nn.Module):
    def __init__(
        self,
        d_in,
        n_classes,
        arch="performer",
        d_model=512,
        n_heads=8,
        n_hidden=1024,
        n_layers=1,
        agg="max",
    ):
        super().__init__()
        assert agg in ["mean", "max"]
        assert arch.lower() in ["performer", "linearattentiontransformer"]

        self.encoder = nn.Linear(d_in, d_model)

        if arch.lower() == "performer":
            self.transformer = Performer(
                dim=d_model, dim_head=64, depth=n_layers, heads=n_heads, causal=False
            )

        elif arch.lower() == "linearattentiontransformer":
            self.transformer = TransformerEncoderBuilder(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=64,
                value_dimensions=64,
                feed_forward_dimensions=n_hidden,
                attention_type="linear",
                activation="gelu",
            ).get()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_classes)
        )

        self.d_model = d_model
        self.agg = agg

        # init weights
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.transformer(x)

        if self.agg == "mean":
            x = torch.mean(x, dim=1)
        elif self.agg == "max":
            x, _ = torch.max(x, dim=1)

        x = self.mlp(x)
        return x


@torch.no_grad()
def evaluate(model, val_dl, device, prefix="", subject=False):
    model.eval()
    logs = {}

    def metrics(logs, target, preds, prefix=""):
        target, preds = target.cpu().numpy(), preds.cpu().numpy()
        logs[f"{prefix}accuracy"] = accuracy_score(target, preds.argmax(-1))
        logs[f"{prefix}f1"] = f1_score(target, preds.argmax(-1))
        logs[f"{prefix}auc"] = roc_auc_score(target, preds[:, 1])
        logs[f"{prefix}aupr"] = average_precision_score(target, preds[:, 1])

    # segment-wise evaluation
    ys, y_preds, ss = [], [], []
    for x, y, s in val_dl:
        ss.append(s)
        ys.append(y)
        y_preds.append(F.softmax(model(x.to(device)), dim=-1).cpu())

    ss = torch.cat(ss)
    ys = torch.cat(ys)
    y_preds = torch.cat(y_preds)
    metrics(logs, ys, y_preds, prefix=prefix)

    if subject:
        # subject-wise evaluation
        ys_subj, y_preds_subj = [], []
        for s in ss.unique():
            for y in ys.unique():
                ys_subj.append(y)
                # average prediction for participant s where label is y
                y_preds_subj.append(y_preds[(ss == s) & (ys == y)].mean(dim=0))

        ys_subj = torch.stack(ys_subj)
        y_preds_subj = torch.stack(y_preds_subj)
        metrics(logs, ys_subj, y_preds_subj, prefix=prefix + "subj_")
    return logs


def train(
    data_path="datasets/cogpilot/CogPilot_MTS_1_4_20_5_norm.pkl",
    arch="performer",
    batch_size=32,
    epochs=2,
    lr=0.001,
    eval_every=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_ds = CogPilotData(data_path, "train")
    val_ds = CogPilotData(data_path, "val")
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=1)
    val_dl = [d for d in val_dl] # pre-cache

    model = Model(d_in=len(train_ds.variables), n_classes=2, arch=arch)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=1,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
        verbose=True,
    )
    steps = 0
    best_metric = 0
    best_params = deepcopy(model.state_dict())
    for epoch in range(epochs):
        for x, y, _ in train_dl:
            if steps % eval_every == 0:
                logs = evaluate(model, val_dl, device, prefix="val_")
                print(f"step={steps}", end="")
                for k, v in logs.items():
                    print(f" {k}={v:.4f} ", end="")
                print()
                if logs["val_f1"] > best_metric:
                    # checkpoint
                    print("SAVED")
                    best_metric = logs["val_f1"]
                    best_params = deepcopy(model.state_dict())

            model.train()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        scheduler.step(logs["val_aupr"])

    model.load_state_dict(best_params)
    test_ds = CogPilotData(data_path, "test")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=1)
    print("TESTING")
    logs = evaluate(model, test_dl, device, subject=True)
    for k, v in logs.items():
        print(f"{k}={v:.4f} ", end="")
    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
