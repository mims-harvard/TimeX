import torch
from sklearn.metrics import f1_score
from txai.models.base_adv_model import compose_adv_model

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
def eval_adv_on_tuple(test_tuple, model, n_classes, mask = None):

    extractor, predictor, _ = model
    # extractor.eval()
    # predictor.eval()

    X, times, y = test_tuple

    pred, mask = compose_adv_model(extractor, predictor)(X, times)
    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    return f1, mask



@torch.no_grad()
def eval_mvts_transformer(test_tuple, model):
    '''
    Returns f1 score
    '''
    model.eval()

    X, times, y = test_tuple

    pred = model(X, times)

    f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

    return f1

# def eval_on_tuple(test_tuple, model, n_classes, by_step = False, sat_output = False, duo = True, mask = None):
#     '''
#     Returns f1 score
#     '''

#     model.eval()

#     X, times, y = test_tuple

#     if by_step:
#         pred = torch.empty((X.shape[1], n_classes)).to(y.get_device())
#         all_attns = []
#         for i in range(X.shape[1]):
#             if sat_output:
#                 pred[i,:], sat_mask, attn = model(X[:,i,:], times[:,i].unsqueeze(-1))
#             else:
#                 pred[i,:], attn = model(X[:,i,:], times[:,i].unsqueeze(-1))

#             all_attns.append(attn)

#     else:
#         if sat_output:
#             if duo:
#                 # sat_scores, joint_mask = model.enc_phi(X, times)
#                 # pred = model.enc_theta(X, times, joint_mask)
#                 pred, sat_scores = model(X, times, mask = mask)
#             else:
#                 pred, sat_mask, all_attns = model(X, times)
#         else:
#             pred, sat_scores = model(X, times)

#     f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

#     return f1, sat_scores

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