import pathlib
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def imp_ft_within_ts(identified_imp, gt_imp):
    """% of the time that for a given timestep, the most important feature has the highest saliency"""
    assert identified_imp.shape == gt_imp.shape
    assert len(identified_imp.shape) == 3, "(samples, fts, ts)"
    assert identified_imp.shape[1] > 1, "(>1 ft required)"

    identified_imp = np.nan_to_num(identified_imp)
    # TODO: Does this assumption about importance make sense
    identified_imp = np.abs(identified_imp)

    std_thresh = 1

    ranked_acc_count = 0
    above_margin_count = 0

    imp_saliency = []
    unimp_saliency = []

    important_ts = np.argwhere(np.max(gt_imp, axis=1))
    for (i, t) in important_ts:
        gt_imp_ft = np.argmax(gt_imp[i, :, t])
        identified_imp_ft = np.argmax(identified_imp[i, :, t])

        imp_saliency.append(identified_imp[i, gt_imp_ft, t])
        unimp_saliency.append(np.delete(identified_imp[i, :, t], gt_imp_ft))

        # Ensure that there is only 1 imp ft
        ranked_gt_fts = np.sort(gt_imp[i, :, t])
        assert ranked_gt_fts[-1] > ranked_gt_fts[-2] == 0, "Should only be 1 imp ft per ts for now"

        # Ensure that the identified most important ft is unique
        ranked_fts = np.sort(identified_imp[i, :, t])
        if ranked_fts[-1] > ranked_fts[-2] and gt_imp_ft == identified_imp_ft:
            # Ranked accuracy
            ranked_acc_count += 1

            # Imp ft is at least n std larger than unimp ft
            above_margin_count += ranked_fts[-1] - ranked_fts[-2] >= np.std(identified_imp[i]) * std_thresh

    print('-----------------------------------------------')
    print('Important features within each timestep metrics')

    imp_ft_acc = ranked_acc_count / len(important_ts)
    print('Ranked accuracy:', imp_ft_acc)

    avg_above_margin = above_margin_count / len(important_ts)
    print(f'Imp ft at least {std_thresh} std > than unimp ft: {avg_above_margin}')

    unimp_saliency, imp_saliency = np.array(unimp_saliency).flatten(), np.array(imp_saliency).flatten()
    unimp_pdf, _ = np.histogram(unimp_saliency, density=True)
    imp_pdf, _ = np.histogram(imp_saliency, density=True)

    print(f'KL div between imp ft and unimp ft in imp ts: {scipy.stats.entropy(imp_pdf, unimp_pdf)}')
    print('-----------------------------------------------')
    return imp_ft_acc, avg_above_margin


def plot_calibration_curve(preds, labels, path):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    fraction_of_pos, mean_predict_value = calibration_curve(labels, preds)
    plt.plot(mean_predict_value, fraction_of_pos)
    plt.savefig(path)


def plot_calibration_curve_from_pytorch(model, test_loader, path, activation=lambda x: x):
    labels = []
    preds = []
    for x, y in test_loader:
        if len(y.shape) == 2:
            y = y[:, -1]
        labels.append(y.detach().numpy())
        preds.append(activation(model(x))[:, 1].cpu().detach().numpy())
    plot_calibration_curve(np.concatenate(preds), np.concatenate(labels), path)
