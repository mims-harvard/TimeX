import torch
import torch.nn.functional as F
import numpy as np

from txai.vis.visualize_cbm1 import visualize
from txai.models.cbmv1 import CBMv1
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_cbmv1
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.evaluation import ground_truth_precision_recall

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_concept_correspondence(concept_scores, labels):

    # Build concept correctness:

    class_correct = []

    for i in [1, 2, 3]:
        focus_mask = (labels == i)

        concept_focus = (concept_scores[focus_mask,:].cpu() > 0.5).float()

        if i == 1:
            simto = torch.tensor([0, 1, 0, 0]).unsqueeze(0).repeat(concept_focus.shape[0], 1)
        elif i == 2:
            simto = torch.tensor([1, 0, 1, 0]).unsqueeze(0).repeat(concept_focus.shape[0], 1)
        elif i == 3:
            simto1 = torch.tensor([0, 1, 1, 0]).unsqueeze(0).repeat(concept_focus.shape[0], 1)
            simto2 = torch.tensor([1, 0, 0, 1]).unsqueeze(0).repeat(concept_focus.shape[0], 1)

        if i == 1 or i == 2:
            sim = F.cosine_similarity((concept_focus > 0.5).float(), simto, dim = 1).mean().item()

        else:
            sim1 = F.cosine_similarity((concept_focus > 0.5).float(), simto1, dim = 1).mean()
            sim2 = F.cosine_similarity((concept_focus > 0.5).float(), simto2, dim = 1).mean()

            sim = max([sim1.item(), sim2.item()])

        class_correct.append(sim)
    return class_correct

def main():

    D = process_Synth(split_no = 1, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    gt_exp = D['gt_exps']

    # Calc statistics for baseline:
    mu = D['train_loader'].X.mean(dim=1)
    std = D['train_loader'].X.std(unbiased = True, dim = 1)

    spath = 'models/Scomb_cbm_mlp_split=1.pt'
    print('Loading model at {}'.format(spath))

    sdict, config = torch.load(spath)

    model = CBMv1(masktoken_kwargs = {'mu': mu, 'std': std}, **config)
    model.to(device)

    model.load_state_dict(sdict)
    if model.distance_method == 'mahalanobis':
        model.load_concept_bank('concept_bank.pt')
    elif model.distance_method == 'centroid':
        model.mu = torch.load('concept_bank_centroid.pt')['mu']

    f1, _ = eval_cbmv1(test, model)
    print('Test F1: {:.4f}'.format(f1))

    yhat, concept_scores, mask_list, _ = model(D['test'][0], D['test'][1])
    yhat = yhat.argmax(dim=-1)

    print('Concept_scores', concept_scores.shape)

    mask = (mask_list[0].bool() | mask_list[1].bool()).float().transpose(0,1)

    total_prec, total_rec, masked_in = ground_truth_precision_recall(mask, gt_exp.transpose(0,1), num_points = 1)

    test_y = D['test'][-1]

    keep = (test_y > 0).cpu().numpy()
    correct = (yhat == test_y).cpu().numpy()[keep]

    # Filter by test_y not equal to 0:
    total_prec, total_rec, masked_in = np.array(total_prec)[keep], np.array(total_rec)[keep], np.array(masked_in)[keep]

    print('Total')
    print('Precision = {:.4f}'.format(np.mean(total_prec)))
    print('Recall = {:.4f}'.format(np.mean(total_rec)))
    print('Masked-in = {:.4f}'.format(np.mean(masked_in)))

    print('Of correct')
    print('Precision = {:.4f}'.format(np.mean(total_prec[correct])))
    print('Recall = {:.4f}'.format(np.mean(total_rec[correct])))
    print('Masked-in = {:.4f}'.format(np.mean(masked_in[correct])))

    sims = eval_concept_correspondence(concept_scores[keep], yhat[keep])
    print('Concept sims scores', sims)


if __name__ == '__main__':
    main()