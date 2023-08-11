import torch
import torch.nn.functional as F
import numpy as np
import ipdb

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4
from txai.utils.cl import in_batch_triplet_sampling
from txai.models.run_model_utils import batch_forwards, batch_forwards_TransformerMVTS
from txai.utils.cl import basic_negative_sampling

from txai.utils.functional import js_divergence

default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 5,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}

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
        lam_label = 1.0,
        clip_norm = True,
        use_scheduler = False,
        wait_for_scheduler = 20,
        scheduler_args = default_scheduler_args,
        selection_criterion = None,
        save_path = None,
        early_stopping = True,
        label_matching = False,
        embedding_matching = True,
        opt_pred_mask = False, # If true, optimizes based on clf_criterion
        opt_pred_mask_to_full_pred = False,
        batch_forward_size = None,
        simclr_training = False,
        num_negatives_simclr = 64,
        max_batch_size_simclr_negs = None,
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

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    dataX, dataT, dataY = train_tuple # Unpack training variables

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_sim_loss = [], [], [], []
        label_sim_list, emb_sim_list = [], []
        for X, times, y, ids in train_loader: # Need negative sampling here

            optimizer.zero_grad()

            # if detect_irreg:
            #     src_mask = (X < 1e-7)
            #     out_dict = model(X, times, captum_input = True)

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

            if sim_criterion is not None:
                if label_matching and embedding_matching:
                    org_embeddings, conc_embeddings = out_dict['all_z']

                    if simclr_training:
                        neg_inds = basic_negative_sampling(X, ids, dataX, num_negatives = num_negatives_simclr)
                        n_inds_flat = neg_inds.flatten()
                        if max_batch_size_simclr_negs is None:
                            neg_embeddings = model.encoder_main.embed(dataX[:,n_inds_flat,:], dataT[:,n_inds_flat], captum_input = False)
                        else:
                            _, neg_embeddings = batch_forwards_TransformerMVTS(model.encoder_main, dataX[:,n_inds_flat,:], dataT[:,n_inds_flat], batch_size = max_batch_size_simclr_negs)

                        # Reshape to split out number of negatives:
                        inds = torch.arange(X.shape[0])
                        #print('is', inds.shape)
                        #print('ne', neg_embeddings.shape)
                        inds_rep = torch.repeat_interleave(inds, num_negatives_simclr)
                        #print(inds_rep)
                        neg_embeddings = torch.stack([neg_embeddings[(inds_rep==j),:] for j in range(X.shape[0])], dim = 0).transpose(1,2)
                        # print('neg_emb', neg_embeddings.shape)
                        # print('c', conc_embeddings.shape)
                        #neg_embeddings = neg_embeddings.view(org_embeddings.shape[0], -1, num_negatives)

                        emb_sim_loss = sim_criterion[0](conc_embeddings, org_embeddings, neg_embeddings)

                    else:
                        if model.ablation_parameters.ptype_assimilation and (not (model.ablation_parameters.side_assimilation)):
                            conc_embeddings = out_dict['ptypes']
                
                        emb_sim_loss = sim_criterion[0](org_embeddings, conc_embeddings)

                        if model.ablation_parameters.side_assimilation:
                            emb_ptype_sim_loss = sim_criterion[0](org_embeddings, out_dict['ptypes'])
                            emb_sim_loss += emb_ptype_sim_loss

                    pred_org = out_dict['pred']
                    pred_mask = out_dict['pred_mask']
                    #print('pre', pred_org)
                    label_sim_loss = sim_criterion[1](pred_mask, pred_org)

                    # print('label', label_sim_loss)
                    # print('emb', emb_sim_loss)
                    # print('----')

                    sim_loss = emb_sim_loss + lam_label * label_sim_loss
                    label_sim_list.append(label_sim_loss.detach().clone().item())
                    emb_sim_list.append(emb_sim_loss.detach().clone().item())

                elif label_matching:
                    pred_org = out_dict['pred']
                    pred_mask = out_dict['pred_mask']
                    sim_loss = sim_criterion(pred_mask, pred_org)
                elif embedding_matching:
                    org_embeddings, conc_embeddings = out_dict['all_z']
                    if model.ablation_parameters.ptype_assimilation:
                        conc_embeddings = out_dict['ptypes']
                    sim_loss = sim_criterion(org_embeddings, conc_embeddings)
                else:
                    raise ValueError('Either label_matching or embedding_matching should be true')
            else:
                sim_loss = torch.tensor(0.0)

            sim_loss = beta_sim * sim_loss
            exp_loss = beta_exp * model.compute_loss(out_dict)
            loss = clf_loss + exp_loss + sim_loss
            # print('---------')
            # print('clf', clf_loss)
            # print('exp', exp_loss)
            # print('sim', sim_loss)
            # print('loss', loss)

            #import ipdb; ipdb.set_trace()

            if clip_norm:
                #print('Clip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # print('loss', loss.item())
            # exit()

            loss.backward()
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.any(torch.isnan(param.grad)):
            #             #print(name, torch.isnan(param.grad).sum())
            #             print(name, param.grad)

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

        if (len(label_sim_list) > 0) and (len(emb_sim_list) > 0):
            sim_s = ['{:.4f}'.format(np.mean(emb_sim_list)), '{:.4f}'.format(np.mean(label_sim_list))]
        else:
            sim_s = f'{sim:.4f}'

        print(f'Epoch: {epoch}: Sparsity = {sparse:.4f} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t CL Loss = {sim_s}')

        # Eval after every epoch
        # Call evaluation function:
        model.eval()
        
        if batch_forward_size is None:
            f1, out = eval_mv4(val_tuple, model)
        else:
            out = batch_forwards(model, val_tuple[0], val_tuple[1], batch_size = 64)
            f1 = 0
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
            if save_path is not None:
                model.save_state(save_path)
            best_epoch = epoch
            print('Save at epoch {}: Metric={:.4f}'.format(epoch, met))

        if use_scheduler and (epoch > wait_for_scheduler):
            scheduler.step(met)

        if (epoch + 1) % 10 == 0:
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')
