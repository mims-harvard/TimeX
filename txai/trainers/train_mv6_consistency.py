import torch
import torch.nn.functional as F
import numpy as np
import ipdb

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4
from txai.utils.cl import in_batch_triplet_sampling

from txai.utils.functional import js_divergence

def train_mv6_consistency(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        # Criterions:
        clf_criterion,
        sim_criterion,
        beta_exp,
        beta_sim,
        train_tuple,
        num_triplets_per_sample = 1,
        clip_norm = True,
        selection_criterion = None,
        save_path = None,
        early_stopping = True,
        label_matching = False,
        embedding_matching = True,
        opt_pred_mask = False, # If true, optimizes based on clf_criterion
        opt_pred_mask_to_full_pred = False,
    ):
    '''
    Args:
        selection_criterion: function w signature f(out, val_tuple)

        if both label_matching and embedding_matching are true, then sim_criterion must be a list of length 2 
            with [embedding_sim, label_sim] functions

    '''
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9

    dataX, dataT, dataY = train_tuple # Unpack training variables

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_sim_loss = [], [], [], []
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

            if opt_pred_mask:
                clf_pred_loss = clf_criterion(out_dict['pred_mask'], y)
                clf_loss += clf_pred_loss
            elif opt_pred_mask_to_full_pred:
                clf_pred_loss = js_divergence(out_dict['pred_mask'].softmax(dim=-1), out_dict['pred'].softmax(dim=-1))
                clf_loss += clf_pred_loss

            # Can do very rough negative sampling here:
            #neg_inds = basic_negative_sampling(X, ids, dataX, num_negatives = num_negatives)
            if sim_criterion is not None:
                if label_matching and embedding_matching:
                    org_embeddings, conc_embeddings = out_dict['all_z']
                    emb_sim_loss = sim_criterion[0](org_embeddings, conc_embeddings)

                    pred_org = out_dict['pred']
                    pred_mask = out_dict['pred_mask']
                    label_sim_loss = sim_criterion[1](pred_mask, pred_org)

                    sim_loss = emb_sim_loss + label_sim_loss

                elif label_matching:
                    pred_org = out_dict['pred']
                    pred_mask = out_dict['pred_mask']
                    sim_loss = sim_criterion(pred_mask, pred_org)
                elif embedding_matching:
                    org_embeddings, conc_embeddings = out_dict['all_z']
                    sim_loss = sim_criterion(org_embeddings, conc_embeddings)
                else:
                    raise ValueError('Either label_matching or embedding_matching should be true')
            else:
                sim_loss = torch.tensor(0.0)

            sim_loss = beta_sim * sim_loss
            exp_loss = beta_exp * model.compute_loss(out_dict)
            loss = clf_loss + exp_loss + sim_loss

            if clip_norm:
                #print('Clip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            cum_sparse.append(((ste_mask).sum() / ste_mask.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append([exp_loss.detach().clone().item()])
            cum_sim_loss.append(sim_loss.detach().item())

            # Positives and negatives for CL:
            # cum_pos.append(pos_loss.mean().detach().cpu().item())
            # cum_neg.append(neg_loss.mean().detach().cpu().item() / num_negatives)

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()
        sim = np.mean(cum_sim_loss)

        print(f'Epoch: {epoch}: Sparsity = {sparse:.4f} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t CL Loss = {sim:.4f}')

        # Eval after every epoch
        # Call evaluation function:
        model.eval()
        f1, out = eval_mv4(val_tuple, model)
        #met = f1 # Copy for use below
        org_embeddings, conc_embeddings = out['all_z']
        #met = 2.0 - sim_criterion(org_embeddings, conc_embeddings)
        #met = (model.score_contrastive(org_embeddings, conc_embeddings)).mean()
        # loss_dict = model.compute_loss(out)
        met = -1.0 * sim

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
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')
