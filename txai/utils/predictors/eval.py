import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from txai.models.run_model_utils import batch_forwards_TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM

@torch.no_grad()
def eval_on_tuple(test_tuple, model, n_classes, mask = None):
    '''
    Returns f1 score
    '''

    model.eval()

    X, times, y = test_tuple

    pred, mask = model(X, times, mask = mask)
    print(pred.shape)

    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    return f1, mask

@torch.no_grad()
def eval_cbmv1(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, concept_scores, masks, logits = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, concept_scores, masks, logits)

@torch.no_grad()
def eval_filter(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, _, masks, logits = model(X, times, captum_input = False)
    # print('masks', masks.shape)
    # print('mask', masks)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, masks, logits)

@torch.no_grad()
def eval_mv2(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, mask_in, ste_mask, smoother_stats, smooth_src = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, mask_in, ste_mask, smoother_stats, smooth_src)

@torch.no_grad()
def eval_mv3(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src)

@torch.no_grad()
def eval_mv3_sim(test_tuple, model):
    model.eval()
    X, times, y = test_tuple
    pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src, zs = model(X, times, captum_input = False)

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, (pred, pred_tilde, mask_in, ste_mask, smoother_stats, smooth_src, zs)

@torch.no_grad()
def eval_mv4(test_tuple, model, masked = False):
    # Also evaluates models above v4
    model.eval()
    X, times, y = test_tuple
    out = model(X, times, captum_input = False)

    if masked:
        pred = out['pred_mask']
    else:
        pred = out['pred']

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, out

@torch.no_grad()
def eval_mv4_idexp(test_tuple, test_tuple_external, model, masked = False):
    # Also evaluates models above v4
    model.eval()
    X, times, y = test_tuple
    out = model(X, times, src_id = test_tuple_external[0], captum_input = False)

    if masked:
        pred = out['pred_mask']
    else:
        pred = out['pred']

    ynp = y.cpu().numpy()
    prednp = pred.argmax(dim=1).detach().cpu().numpy()
    f1 = f1_score(ynp, prednp, average='macro')
    return f1, out

@torch.no_grad()
def eval_mvts_transformer(test_tuple, model, batch_size = None, auprc = False, auroc = False):
    '''
    Returns f1 score
    '''
    model.eval()

    X, times, y = test_tuple

    if batch_size is not None:
        if isinstance(model, CNN) or isinstance(model, LSTM):
            pred = torch.cat(
                [model(xb, tb) for xb, tb in zip(torch.split(X, batch_size, dim=1), torch.split(times, batch_size, dim=1))],
                dim=0
            )
        else: 
            pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size = batch_size)
    else:
        pred = model(X, times)

    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    pred_prob = pred.softmax(dim=-1).detach().cpu().numpy()
    # if pred_prob.shape[-1] == 2:
    #     pred_prob = pred_prob[:,1]

    yc = y.cpu().numpy()
    one_hot_y = np.zeros((yc.shape[0], yc.max() + 1))
    for i, yi in enumerate(yc):
        one_hot_y[i,yi] = 1

    if auprc:
        auprc_val = average_precision_score(one_hot_y, pred_prob, average = 'macro')
    if auroc:
        auroc_val = roc_auc_score(one_hot_y, pred_prob, average = 'macro', multi_class = 'ovr')

    if auprc and auroc:
        return f1, auprc_val, auroc_val
    elif auprc:
        return f1, auprc_val
    elif auroc:
        return f1, auroc_val
    else:
        return f1

@torch.no_grad()
def eval_and_select(X, times, model):
    '''
    Duo version
    Designed for SAT time mask selector
    Assumed X is size (T,d), times is (T,1) or (T,)
    '''

    model.eval()

    if len(times.shape) < 2:
        # Reshape times
        times = times.unsqueeze(-1)

    if len(X.shape) < 3:
        # Unsqueeze to batch
        X = X.unsqueeze(1)

    pred, mask = model(X, times, captum_input = False)

    #mask = (sat_scores > 0.5).squeeze()
    #print('mask size', mask.shape, mask.sum())
    #selected_X = X[mask,:,:]
    selected_X = model.apply_mask(X, times, mask = mask)

    return selected_X, pred.argmax(dim=1).detach().cpu()

@torch.no_grad()
def eval_and_select_nonduo(X, times, model):
    '''
    Designed for SAT time mask selector
    Assumed X is size (T,d), times is (T,1) or (T,)
    '''

    model.eval()

    if len(times.shape) < 2:
        # Reshape times
        times = times.unsqueeze(-1)

    if len(X.shape) < 3:
        # Unsqueeze to batch
        X = X.unsqueeze(1)

    pred, sat_mask, all_attns = model(X, times)

    mask = (sat_mask > 0.5).squeeze()
    print('mask size', mask.shape, mask.sum())
    selected_X = X[mask,:,:]

    return selected_X, pred.argmax(dim=1).detach().cpu()