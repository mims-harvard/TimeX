import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from txai.smoother import smoother
from txai.utils.predictors.loss import Poly1CrossEntropyLoss, L1Loss_permask, PairwiseDecorrelation
from txai.trainers.train_cbmv1 import train_cbmv1
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.cbmv1 import CBMv1
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_cbmv1
from txai.synth_data.simple_spike import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    D = process_Synth(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
    )
    model.load_state_dict(torch.load('../seqcombsingle/models/Scomb_transformer_split={}.pt'.format(1)))
    model.to(device)

    # Iterate over test data:

    samp = np.random.choice(np.arange(test[0].shape[1]))

    sampX, samptime, sampy = test[0][:,samp,:], test[1][:,samp], test[2][samp]
    sampX, samptime, sampy = sampX.unsqueeze(1), samptime.unsqueeze(1), sampy.unsqueeze(0)

    print('sampX', sampX.shape)
    print('time', samptime.shape)
    print('y', sampy.shape)

    model.eval()
    with torch.no_grad():
        pred = model(sampX, samptime)[0]

        print(pred)

        pvals = np.linspace(0.01, 10, num=100)

        sims = []
        clf = []

        for p in pvals:
            transX = smoother(sampX, samptime, p = p)
            #print('TransX', transX.shape)
            pred_i = model(transX, samptime)[0]
            sim = F.cosine_similarity(pred_i, pred, dim = 0).detach().clone().item()
            clf.append(pred_i.argmax(dim=0).detach().clone().item())
            sims.append(sim)

    print(sims)

    #plt.plot(pvals, sims)
    plt.plot(pvals, sims)
    plt.show()

    Xnp = sampX.squeeze().detach().clone().cpu().numpy()
    Xnp_smooth = smoother(sampX, samptime, p = 5).squeeze().detach().clone().cpu().numpy()
    plt.plot(Xnp)
    plt.plot(Xnp_smooth)
    plt.show()

if __name__ == '__main__':
    main()