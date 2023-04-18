import torch
import torch.nn.functional as F
import numpy as np
import ipdb

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4
from txai.utils.cl import basic_negative_sampling

def stat_string(smoother_stats):

    alpha, beta, thresh, p = smoother_stats
    
    alphamean, betamean = alpha.detach().cpu().mean(dim=0), beta.detach().cpu().mean(dim=0)
    threshmean, pmean = thresh.detach().cpu().mean(dim=0), p.detach().cpu().mean(dim=0)
    s = 'a = {:.4f}, b = {:.4f}, thr = {:.4f}, p = {:.4f}'.format(alphamean.item(), betamean.item(), threshmean.item(), pmean.item())
    return s


def train_mv5(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        # Criterions:
        clf_criterion,
        exp_criterion,
        sim_criterion,
        beta_exp,
        beta_sim,
        train_tuple,
        num_negatives = 16,
        clip_norm = True,
        selection_criterion = None,
        save_path = None,
        early_stopping = True,
        loss_uses_dict = False,
    ):
    '''
    Args:
        selection_criterion: function w signature f(out, val_tuple)
    '''
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9

    dataX, dataT, dataY = train_tuple # Unpack training variables

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_logit_loss = [], [], [], []
        cum_pos, cum_neg = [], []
        for X, times, y, ids in train_loader: # Need negative sampling here

            optimizer.zero_grad()

            #print('ids', ids)

            #ipdb.set_trace()

            #out, out_tilde, masks, ste_mask, smoother_stats, smooth_src, (z, z_tilde) = model(X, times, captum_input = True)
            out_dict = model(X, times, captum_input = True)
            out = out_dict['pred']
            ste_mask = out_dict['ste_mask']

            if out.isnan().sum() > 0:
                # Exits if nan's are found
                print('out', out.isnan().sum())
                exit()

            clf_loss = clf_criterion(out, y)

            total_eloss_list = []
            # All explanation criterion operate directly on the smoother statistics
            masks, smoother_stats = out_dict['mask_logits'], out_dict['smoother_stats']
            if exp_criterion is not None:
                if loss_uses_dict:
                    print('is nan', torch.isnan(masks).sum())
                    exp_loss = exp_criterion(X, times, out_dict)
                    eloss_list = [exp_loss.detach().clone().item()]
                    print('el', exp_loss.detach().clone().item())
                else:
                    exp_loss, eloss_list = exp_criterion_eval_smoothers(X, times, masks, smoother_stats, beta_exp, exp_criterion)
            else:
                exp_loss = 0
                eloss_list = [0]

            # Can do very rough negative sampling here:
            neg_inds = basic_negative_sampling(X, ids, dataX, num_negatives = num_negatives)
            #print('neg_inds', neg_mask.sum())
            pos_embeddings, mask_embeddings = out_dict['all_z']

            n_inds_flat = neg_inds.flatten()
            neg_embeddings = model.encoder_main.embed(dataX[:,n_inds_flat,:], dataT[:,n_inds_flat], captum_input = False)
            # Reshape to split out number of negatives:
            neg_embeddings = neg_embeddings.view(mask_embeddings.shape[0], -1, num_negatives)
            # print('neg embeddings', neg_embeddings.shape)
            # print('mask_embeddings', mask_embeddings.shape)
            #exit()
            
            sim_loss, pos_loss, neg_loss = sim_criterion(
                embeddings = mask_embeddings,
                positives = pos_embeddings.unsqueeze(-1), # Must unsqueeze to be equivalent to 1 sample for positives
                negatives = neg_embeddings,
                get_all_scores = True
            )
            sim_loss = beta_sim * sim_loss

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

            # Positives and negatives for CL:
            cum_pos.append(pos_loss.mean().detach().cpu().item())
            cum_neg.append(neg_loss.mean().detach().cpu().item() / num_negatives)

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()
        logit = np.mean(cum_logit_loss)

        print(f'Epoch: {epoch}: Sparsity = {sparse:.4f} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t CL Loss = {logit:.4f} (pos={np.mean(cum_pos):.3f}, neg={np.mean(cum_neg):.3f})')

        # Eval after every epoch
        # Call evaluation function:
        f1, out = eval_mv4(val_tuple, model)
        met = f1 # Copy for use below

        ste_mask = out['ste_mask']
        sparse = ste_mask.mean().item()

        cond = not early_stopping
        if early_stopping:
            if selection_criterion is not None:
                met = selection_criterion(out, val_tuple)
                cond = (met > best_val_metric)
            else:
                # Early stopping procedure:
                # Don't recalculate met
                cond = (met > best_val_metric)
        if cond:
            best_val_metric = met
            model.save_state(save_path)
            best_epoch = epoch
            print('Save at epoch {}: Metric={:.4f}'.format(epoch, met))

        if (epoch + 1) % 10 == 0:
            smoother_stats = out['smoother_stats']
            if isinstance(smoother_stats, tuple):
                sstring = stat_string(smoother_stats)
            else:
                sstring = '{:.4f}'.format(smoother_stats.detach().cpu().mean(dim=0).item())
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}, Stats: {sstring}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')
