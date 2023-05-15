from argparse import ArgumentParser
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_Epilepsy

def dict_prefix(d, prefix=""):
    """add prefix to all dictionary keys"""
    return {prefix + k: v for k, v in d.items()}


def dict_to_string(d):
    """convert dictionary to string, ignoring elements that aren't floats"""
    return " ".join([f"{k}: {v:.3f}" for k, v in d.items() if isinstance(v, float)])


class Encoder(nn.Module):
    def __init__(self, in_channels, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1),  # 2x dim for mu, sigma
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        # batch, time, channels -> batch, channels, time
        x = x.permute(0, 2, 1)
        return self.encoder(x)

class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        
    def forward(self, x):
        return torch.matmul(x, torch.exp(self.weight)) + self.bias

class Decoder(nn.Module):
    def __init__(self, in_dim, out_channels=1, dim=128):
        super().__init__()
        self.dim = dim
        # original length of 178, with 3 stride=2 max pools, so this should be
        # the length of representations before the adaptive average pool
        self.length = 178 // 2 // 2 // 2
        self.adapt_inv = nn.Linear(in_dim, dim * self.length)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(dim, out_channels, kernel_size=7, padding=4),
        )

    def forward(self, x):
        x = self.adapt_inv(x)
        x = x.view(x.shape[0], self.dim, self.length)
        x = self.decoder(x)
        # batch, channels, time -> batch, time, channels
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, channels, dim, n_classes, n_prototypes):
        super().__init__()
        self.dim = dim
        assert (
            n_prototypes % n_classes == 0
        ), "n_prototypes must be divisible by n_classes"
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.encoder = Encoder(channels, dim)
        self.decoder = Decoder(dim, channels)
        self.final = nn.Linear(self.n_prototypes, self.n_classes, bias=False)

        prototype_class_identity = torch.zeros(self.n_prototypes, self.n_classes)
        self.n_prototypes_per_class = self.n_prototypes // self.n_classes
        print("n_prototypes_per_class", self.n_prototypes_per_class)
        for j in range(self.n_prototypes):
            prototype_class_identity[j, j // self.n_prototypes_per_class] = 1
        self.register_buffer("prototype_class_identity", prototype_class_identity)

        self.prototype_vectors = nn.Parameter(
            torch.randn(self.n_prototypes, self.dim), requires_grad=True
        )
        self.ones = nn.Parameter(
            torch.ones(self.n_prototypes, self.dim), requires_grad=False
        )

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + 1e-4))

    def calc_sim_scores(self, z):
        d = torch.cdist(z, self.prototype_vectors, p=2)  ## Batch size x prototypes
        sim_scores = self.distance_2_similarity(d)
        return sim_scores

    def kl_divergence_nearest(self, mu, log_var, nearest_pt, sim_scores):
        kl_loss = torch.zeros_like(sim_scores)
        for i in range(self.n_prototypes_per_class):
            p = torch.distributions.Normal(mu, torch.exp(log_var / 2))
            p_v = self.prototype_vectors[nearest_pt[:, i], :]
            q = torch.distributions.Normal(p_v, torch.ones_like(p_v))
            kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1)
            kl_loss[torch.arange(sim_scores.shape[0]), nearest_pt[:, i]] = kl
        kl_loss = kl_loss * sim_scores
        mask = kl_loss > 0
        kl_loss = torch.sum(kl_loss, dim=1) / (torch.sum(sim_scores * mask, dim=1))
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def get_prototype_images(self):
        p_decoded = self.decoder(self.prototype_vectors)
        return p_decoded

    def ortho_loss(self):
        s_loss = 0
        for k in range(self.n_classes):
            p_k = self.prototype_vectors[
                k * self.n_prototypes_per_class : (k + 1) * self.n_prototypes_per_class,
                :,
            ]
            p_k_mean = torch.mean(p_k, dim=0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(p_k.device))
            s_loss += torch.norm(s_matrix, p=2)
        return s_loss / self.n_classes

    def forward(self, x, y, train=True):
        features = self.encoder(x)

        mu, log_var = features[:, self.dim :], features[:, : self.dim]

        # reparameterize
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(mu)
        if train:
            z = mu + std * eps
        else:
            z = mu

        sim_scores = self.calc_sim_scores(z)

        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, y])

        index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(
            as_tuple=True
        )[1]
        index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(
            x.shape[0], self.n_prototypes_per_class
        )

        kl_loss = self.kl_divergence_nearest(
            mu, log_var, index_prototypes_of_correct_class, sim_scores
        )

        out = self.final(sim_scores)

        decoded = self.decoder(z)

        ortho_loss = self.ortho_loss()

        ce_loss = F.cross_entropy(out, y)
        recons_loss = F.mse_loss(decoded, x)

        loss = 1e2 * ce_loss + recons_loss + kl_loss + ortho_loss

        return {
            "predictions": out,
            "reconstruction": decoded,
            "loss_ortho": ortho_loss,
            "loss_classification": ce_loss,
            "loss_reconstruction": recons_loss,
            "loss_kl": kl_loss,
        }, loss


# run with `python -m pytest protovae.py`
def test_encoder_decoder():
    seq_len = 178
    dim = 16

    encoder = Encoder(in_channels=1, dim=dim)
    decoder = Decoder(in_dim=dim, out_channels=1)
    x = torch.randn(8, seq_len, 1)  # batch, seq, channels
    features = encoder(x)
    mu, std = features[:, dim:], features[:, :dim]
    x_hat = decoder(mu)
    assert x_hat.shape == x.shape


def test_model():
    seq_len = 178
    dim = 16

    x = torch.randn(8, seq_len, 1)  # batch, seq, channels
    y = torch.randint(0, 3, size=(8,))
    model = Model(channels=1, dim=16, n_classes=4, n_prototypes=40)
    out, loss = model(x, y)
    assert out["predictions"].shape == (8, 4)
    assert out["reconstruction"].shape == (8, seq_len, 1)


def train(
    epochs=100,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    n_prototypes=6,
    base_path="datasets/drive/datasets_and_models/Epilepsy",
    early_stop=True,
):
    torch.manual_seed(0)

    train_epi, val_epi, test_epi = process_Epilepsy(split_no=1, device=device, base_path=base_path)
    train_dataset = EpiDataset(train_epi.X, train_epi.time, train_epi.y)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # time, batch, channels -> batch, time, channels
    valid_ds = (val_epi.X.permute(1, 0, 2), val_epi.y)
    test_ds = (test_epi.X.permute(1, 0, 2), test_epi.y)

    model = Model(
        channels=1, dim=128, n_classes=2, n_prototypes=n_prototypes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    @torch.no_grad()
    def test(dataset, prefix="val_"):
        model.eval()
        x, y = dataset
        output, _ = model(x.float().to(device), y.long().to(device), train=False)
        preds = F.softmax(output["predictions"], dim=-1).cpu()
        acc = accuracy_score(y.cpu().numpy(), preds.argmax(-1).numpy())
        f1 = f1_score(y.cpu().numpy(), preds.argmax(-1).numpy(), average="macro")
        return dict_prefix(
            {
                "acc": acc,
                "f1": f1,
                "loss_ortho": output["loss_ortho"].item(),
                "loss_classification": output["loss_classification"].item(),
                "loss_kl": output["loss_kl"].item(),
                "loss_reconstruction": output["loss_reconstruction"].item(),
                "x": x.cpu().numpy(),
                "reconstruction": output["reconstruction"].cpu().numpy(),
            },
            prefix,
        )

    best_acc = -float("inf")
    best_params = None

    for epoch in range(epochs):
        model.train()
        for x, _, y in train_dl:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = test(valid_ds, prefix="val_")
        print("epoch:", epoch, dict_to_string(metrics))
        if metrics["val_f1"] > best_acc:
            best_params = copy.deepcopy(model.state_dict())
            best_acc = metrics["val_f1"]

    if early_stop:
        model.load_state_dict(best_params)

    torch.save(model.state_dict(), f"models/epilepsy_protovae_prototypes={n_prototypes}.pt")

    metrics = test(test_ds, prefix="test_")
    print(dict_to_string(metrics))

    # plot examples of reconstructions
    n_examples = 5
    for i in range(n_examples):
        plt.plot(metrics["test_x"][i, :, 0], label="x")
        plt.plot(metrics["test_reconstruction"][i, :, 0], label="reconstruction")
        plt.legend()
        plt.title(f"Epilepsy: Test[{i}] ProtoVAE")
        plt.savefig(f"figures/protovae_eg_{i}_epilepsy.png")
        plt.close("all")

    fig, axs = plt.subplots(int(model.n_prototypes_per_class), int(model.n_classes), figsize=(10, 12))

    with torch.no_grad():
        for k in range(model.n_classes):
            p_k = model.prototype_vectors[
                k * model.n_prototypes_per_class : (k + 1) * model.n_prototypes_per_class,
                :,
            ]
            for j in range(model.n_prototypes_per_class):
                decoded_proto = model.decoder(p_k[j].unsqueeze(0))[0]
                if j == 0:  # set title on first example
                    axs[j, k].set_title(f"class={k}")
                axs[j, k].plot(decoded_proto.squeeze().cpu().numpy())
                # axs[i, j].axis("off")
                axs[j, k].xaxis.set_ticks([])
                axs[j, k].yaxis.set_ticks([])

        fig.suptitle("Prototypes")
        fig.tight_layout()
        plt.savefig(f"figures/prototypes_epilepsy.png")
        plt.close("all")
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n_prototypes", type=int, default=16)
    parser.add_argument("--data_path", default="datasets/drive/datasets_and_models/Epilepsy/")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        n_prototypes=args.n_prototypes,
        base_path=args.data_path,
    )