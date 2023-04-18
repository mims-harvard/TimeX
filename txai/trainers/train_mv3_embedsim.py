import torch
import torch.nn.functional as F
import numpy as np
import ipdb

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv3_sim

def stat_string(smoother_stats):

    alpha, beta, thresh, p = smoother_stats
    
    alphamean, betamean = alpha.detach().cpu().mean(dim=0), beta.detach().cpu().mean(dim=0)
    threshmean, pmean = thresh.detach().cpu().mean(dim=0), p.detach().cpu().mean(dim=0)
    s = 'a = {:.4f}, b = {:.4f}, thr = {:.4f}, p = {:.4f}'.format(alphamean.item(), betamean.item(), threshmean.item(), pmean.item())
    return s


def train_mv3_embedsim(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        clf_criterion,
        exp_criterion,
        sim_criterion,
        beta_exp,
        beta_sim,
        clip_norm = True,
        selection_criterion = None,
        save_path = None,
    ):
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_logit_loss = [], [], [], []
        for X, times, y in train_loader:

            optimizer.zero_grad()

            #ipdb.set_trace()

            out, out_tilde, masks, ste_mask, smoother_stats, smooth_src, (z, z_tilde) = model(X, times, captum_input = True)

            if out.isnan().sum() > 0:
                print('out', out.isnan().sum())
                exit()

            clf_loss = clf_criterion(out, y)

            total_eloss_list = []
            # All explanation criterion operate directly on the smoother statistics
            exp_loss, eloss_list = exp_criterion_eval_smoothers(X, times, masks, smoother_stats, beta_exp, exp_criterion)
            #logit_loss = beta_logit * logit_criterion(F.log_softmax(out, dim=-1), F.softmax(out_tilde, dim=-1))
            sim_loss = beta_sim * -1.0 * sim_criterion(z, z_tilde)
            sim_loss = sim_loss.mean()

            #exp_loss /= len(logits) # Normalize out with respect to number of masks in model

            loss = clf_loss + exp_loss + sim_loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            cum_sparse.append(((ste_mask).sum() / ste_mask.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append([eloss_list])
            cum_logit_loss.append(sim_loss.detach().item())

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()
        logit = np.mean(cum_logit_loss)

        print(f'Epoch: {epoch}: Sparsity = {sparse} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t Logit Loss = {logit:.4f}')
        #exit()

        # Eval after every epoch
        # Call evaluation function:
        f1, (pred, pred_tilde, masks, ste_mask, smoother_stats, smooth_src, _) = eval_mv3_sim(val_tuple, model)
        # exp_loss, _ = exp_criterion_eval_smoothers(val_tuple[0].transpose(0,1), val_tuple[1].transpose(0,1), masks, smoother_stats, beta_exp, exp_criterion)
        # logit_loss = beta_sim * sim_criterion(F.log_softmax(pred, dim=-1), F.softmax(pred_tilde, dim=-1))
        # total_loss = exp_loss + logit_loss
        sparse = ste_mask.mean().item()

        # Early stopping procedure:
        if f1 > best_val_metric:
            best_val_metric = f1
            model.save_state(save_path)
            best_epoch = epoch
        #print('Save at epoch', epoch)

        if (epoch + 1) % 10 == 0:
            sstring = stat_string(smoother_stats)
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}, Stats: {sstring}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')
