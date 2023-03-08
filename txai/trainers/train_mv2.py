import torch
import numpy as np
import ipdb

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv2

def stat_string(smoother_stats):

    alpha, beta, thresh, p = smoother_stats
    
    alphamean, betamean = alpha.detach().cpu().mean(dim=0), beta.detach().cpu().mean(dim=0)
    threshmean, pmean = thresh.detach().cpu().mean(dim=0), p.detach().cpu().mean(dim=0)
    s = 'a = {:.4f}, b = {:.4f}, thr = {:.4f}, p = {:.4f}'.format(alphamean.item(), betamean.item(), threshmean.item(), pmean.item())
    return s


def train_mv2(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        clf_criterion,
        exp_criterion,
        beta,
        clip_norm = True,
        selection_criterion = None,
        save_path = None,
    ):
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss = [], [], []
        for X, times, y in train_loader:

            optimizer.zero_grad()

            #ipdb.set_trace()

            out, masks, smoother_stats, smooth_src = model(X, times, captum_input = True)

            if out.isnan().sum() > 0:
                print('out', out.isnan().sum())
                exit()

            clf_loss = clf_criterion(out, y)

            total_eloss_list = []
            # All explanation criterion operate directly on the smoother statistics
            exp_loss, eloss_list = exp_criterion_eval_smoothers(X, times, smoother_stats, beta, exp_criterion)

            #exp_loss /= len(logits) # Normalize out with respect to number of masks in model

            loss = clf_loss + exp_loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            cum_sparse.append(((masks > 0).sum() / masks.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append([eloss_list])

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()

        print(f'Epoch: {epoch}: Sparsity = {sparse} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f}')
        #exit()

        # Eval after every epoch
        # Call evaluation function:
        f1, (pred, masks, smoother_stats, smooth_src) = eval_mv2(val_tuple, model)

        # Early stopping procedure:
        if f1 > best_val_metric:
            best_val_metric = f1
            model.save_state(save_path)
            best_epoch = epoch


        if (epoch + 1) % 10 == 0:
            valsparse = '{:.4f}'.format(masks.mean().item())
            sstring = stat_string(smoother_stats)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}, Stats: {sstring}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')
