import torch
import numpy as np
import matplotlib.pyplot as plt
from txai.vis.visualize_cbm1 import get_x_mask_borders

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
        pred, mask_in, smoother_stats, smooth_src = model(sampX, samp_times, captum_input = False)
    pred = pred.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)
    print('mask_in', mask_in.shape)
    print('smoother_stats', smoother_stats)
    #print(mask_in)

    mask = (mask_in > 1e-9)

    title_format1 = 'y={:1d}, yhat={:1d}'

    #exit()

    for i in range(n): # Iterate over samples

        # fit lots of info into the title
        yi = samp_y[i].item()
        pi = pred[i].item()
        ax[0,i].set_title(title_format1.format(yi, pi))
        ax[1,i].set_title('Smoothed')


        ax[0,i].plot(x_range, sampX[:,i,:].cpu().numpy(), color = 'black')
        ax[1,i].plot(x_range, smooth_src[:,i,:].cpu().numpy(), color = 'black')

        block_inds = get_x_mask_borders(mask = mask[i,:])

        for k in range(len(block_inds)):
            #print(block_inds[k])
            ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
            ax[1,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

    if show:
        plt.show()





