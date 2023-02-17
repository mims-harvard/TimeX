import torch
import numpy as np
import matplotlib.pyplot as plt

def get_x_mask_borders(mask):
    nz = mask.nonzero(as_tuple=True)[0].tolist()
    # Get contiguous:
    nz_inds = [(nz[i], nz[i+1]) for i in range(len(nz) - 1)]
    return nz_inds

def visualize(model, test_tup, n = 3, per_class = False, class_num = None, show = True):
    # Quick function to visualize some samples in test_tup
    # FOR NOW, assume only 2 masks, 2 concepts

    X, times, y = test_tup

    assert not per_class or (class_num is None), "You can't set per_class=True and class_num not None"

    choices = np.arange(X.shape[1])
    if class_num is not None:
        choices = choices[(y == class_num).cpu().numpy()]
    inds = torch.from_numpy(np.random.choice(choices, size = (n,), replace = False)).long()
    fig, ax = plt.subplots(2, n, sharex = True)

    sampX, samp_times, samp_y = X[:,inds,:], times[:,inds], y[inds]
    x_range = torch.arange(sampX.shape[0])

    model.eval()
    with torch.no_grad():
        pred, concept_score, masks, logits = model(sampX, samp_times)
    pred = pred.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)
    print('cscore', concept_score.shape)
    print('masks len({})'.format(len(masks)), masks[0].shape)
    print('logits len({})'.format(len(logits)), logits[0].shape)

    title_format1 = 'y={:1d}, yhat={:1d}, mask1, s_inc=({:.4f})'
    title_format2 = 'mask2, s_inc=({:.4f})'

    for i in range(n):

        # fit lots of info into the title
        yi = samp_y[i].item()
        pi = pred[i].item()
        c1i = concept_score[i,0].item()
        c2i = concept_score[i,2].item()
        ax[0,i].set_title(title_format1.format(yi, pi, c1i))
        ax[1,i].set_title(title_format2.format(c2i))

        for j in range(2):

            ax[j,i].plot(x_range, sampX[:,i,:].cpu().numpy(), color = 'black')

            block_inds = get_x_mask_borders(mask = masks[j][:,i,:])

            for k in range(len(block_inds)):
                print(block_inds[k])
                ax[j,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

    if show:
        plt.show()





