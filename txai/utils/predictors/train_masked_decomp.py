import sys, os
import torch
import numpy as np
from tqdm import trange, tqdm

from .train_transformer import default_scheduler_args
from txai.utils.masking import random_time_mask

from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
from loss import Poly1CrossEntropyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exp_criterion_evaluation(mask: torch.Tensor, beta: float, exp_criterion: torch.nn.Module):

    if not (isinstance(beta, torch.Tensor) and (isinstance(exp_criterion, list))):
        l = exp_criterion(mask)
        return beta * l, [l.item()]

    # Else, need list-based evaluation
    llist = []
    for i in range(len(beta)):
        
        l = exp_criterion[i](mask)
        llist.append(l.item())

        if i == 0:
            lsum = beta[i] * l
        else:
            lsum += beta[i] * l

    return lsum, llist

def train_masked_decomp(
        extractor,
        predictor,
        train_loader, 
        val_tuple, # Must be (X, times, y) all in one tensor for each 
        num_epochs,
        clf_criterion,
        exp_criterion,
        optimizer,
        adv_optimizer = None,
        adv_predictor = None,
        pretrain_optimizer = None,
        adv_update_freq = 5, 
        delay_phi = 0,
        train_random = None,
        use_zero_vec_perturb = False,
        beta = 1,
        gamma = 1,
        standardize = False,
        save_path = None,
        show_mask_sparsity = False,
        scheduler_args = default_scheduler_args,
        use_scheduler = True,
        clip_norm = False,
        select_by_loss = True,
        gamma_decay_beta = None,
        extractor_loss_lower_bound = 1e-4,
        add_beta_factor = 5,
        optimize_separate = False,
        use_full_samples = False,
        selection_criterion = None,
        performance_scoring = 'f1', # TODO: implement
        pretrain_separate = True): 
    '''
    Loader should output (B, d, T) - in style of captum input

    NOTE: Model needs to not return original attention (i.e. return_org_attn = False)

    TODO: Describe each parameter

    clf_criterion: Loss function for classification
        - Must have forward signature (logits, labels)
    exp_criterion: Loss function for explainer
        - Must have forward signature (attention)
    delay_phi: Number of epochs for which to train prediction encoder before training explainer
    train_random (bool, optional): If provided, use random masking on the inputs when 
        pre-training delay_phi.
    '''

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_args
        )

    if save_path is None:
        save_path = 'tmp.pt'

    if pretrain_optimizer is None:
        pretrain_optimizer = optimizer

    ADV = (adv_predictor is not None)

    pretrain_train_loss, pretrain_val_f1 = [], []

    if delay_phi > 0:
        for p in extractor.parameters():
            p.requires_grad = False

    for epoch in trange(delay_phi):
        # Train only classifier

        predictor.train()

        for X, times, y in train_loader:

            pretrain_optimizer.zero_grad()
            out = predictor(X, times, captum_input = True) 
            pred_loss = clf_criterion(out, y)
            pred_loss.backward()
            pretrain_optimizer.step()

            if train_random is not None:
                # Choose some random number of samples in the batch:
                num_perturb = torch.randint(1,X.shape[0], size = (1,))[0].item() # Number of samples to perturb
                to_perturb = torch.randperm(X.shape[0])[:num_perturb] # Which samples should be perturbed

                # Make random mask based on train_random rate
                msize = (X.shape[1], X.shape[2])
                masked_samples, masked_times = [], []
                ymask = torch.zeros(num_perturb).long().to(device)
                i = 0
                for t in to_perturb:
                    mask_trend = random_time_mask(rate = (1 - train_random), size = msize).to(device).unsqueeze(0)
                    mask_seasonal = random_time_mask(rate = (1 - train_random), size = msize).to(device).unsqueeze(0)
                    #print('maskj size', mask.shape)

                    masked_X, masked_T = extractor.apply_masks(X[t,:,:].unsqueeze(0), times[t,:].unsqueeze(0), 
                        mask_trend, mask_seasonal, captum_input = True)

                    #new_sample = (mask) * X[t] + (1 - mask) * tofill
                    masked_samples.append(masked_X.transpose(0,1))
                    masked_times.append(masked_T.transpose(0,1))

                    ymask[i] = y[t]
                    i += 1

                Xmask = torch.cat(masked_samples, dim = 1).to(device)
                Tmask = torch.cat(masked_times, dim = 1).to(device)
                pretrain_optimizer.zero_grad()
                out = predictor(Xmask, Tmask, captum_input = False) 
                pred_loss = clf_criterion(out, ymask)
                pred_loss.backward()
                pretrain_optimizer.step()

        # model.eval()
        # with torch.no_grad():
        #     # Use no mask on the model output
        #     X, times, y = val_tuple
        #     pred = model.enc_theta(X, times, None, captum_input = False)

        #     # Calculate F1:
        #     f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

        #     # if use_scheduler:
        #     #     scheduler.step(f1) # Step the scheduler

        #     pretrain_val_f1.append(f1)

        # # Don't use scheduler on delay_phi

        # if ((epoch + 1) % 10 == 0) or (epoch == (delay_phi - 1)): # Print progress:
        #     print('Pretrain Epoch {}, Train Loss = {:.4f}, Val F1 = {:.4f}'.format(epoch + 1, loss.item(), f1))

    if delay_phi > 0:
        # Unfreeze extractor
        for p in extractor.parameters():
            p.requires_grad = True

        # Copy weights over to adv_predictor:
        if ADV:
            adv_predictor.load_state_dict(predictor.state_dict())


    train_loss, val_loss, val_auc, val_select, adv_losses = [], [], [], [], []
    best_epoch = 0
    min_val_loss = -1e10
    for epoch in range(num_epochs):
        
        # Decays beta if needed:
        if gamma_decay_beta:
            beta += (add_beta_factor/num_epochs) * beta 

        # Flip all to train:
        if ADV:
            adv_predictor.train()
        extractor.train()
        predictor.train()

        # Decide on update frequencies:
        adv_flag = ((epoch + 1) % adv_update_freq) == 0
        if ADV:
            if adv_flag:
                # Updating adversarial agent:
                for p in adv_predictor.parameters():
                    p.requires_grad = True

                for p in extractor.parameters():
                    p.requires_grad = False

                for p in predictor.parameters():
                    p.requires_grad = False


            else:
                # Updating extractor and predictor pipeline
                for p in adv_predictor.parameters():
                    p.requires_grad = False

                for p in extractor.parameters():
                    p.requires_grad = True

                for p in predictor.parameters():
                    p.requires_grad = True

        # Train:
        cum_info_loss_T, cum_info_loss_S, cum_clf_loss = [], [], []
        cum_adv, cum_adv_opt = [], []
        for X, times, y in train_loader:
            #mask, logits, joint_mask = extractor(X, times, captum_input = True)
            masks, logits, joint_masks = extractor(X, times, captum_input = True)
            mask_trend, mask_seasonal = masks
            logits_trend, logits_seasonal = logits
            jm_trend, jm_seasonal = joint_masks
            masked_src, masked_times = extractor.apply_masks(X, times, jm_trend, jm_seasonal, captum_input = True)
            #masked_src, masked_times, masked_src_tilde, masked_times_tilde = extractor.apply_masks(X, times, joint_mask, captum_input = True)

            if adv_flag and ADV:
                adv_optimizer.zero_grad()
                # ---------------------------
                # Optimize adversarial net:
                # ---------------------------
                output = adv_predictor(masked_src_tilde, masked_times_tilde, captum_input = True)
                adv_loss = clf_criterion(output, y)
                adv_loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(adv_predictor.parameters(), 1.0)
                adv_optimizer.step()

            else:
                optimizer.zero_grad()
                # ---------------------------
                # Optimize full architecture:
                # ---------------------------
                # Step 1: Classification loss:
                output = predictor(masked_src, masked_times, captum_input = True)
                clf_loss = clf_criterion(output, y)

                # Step 2: Adversarial loss:
                if ADV:
                    out_tilde = adv_predictor(masked_src_tilde, masked_times_tilde, captum_input = True)
                    out_tilde_probs = out_tilde.softmax(dim=1)
                    # Maximize entropy of logits, i.e. confusion of the classifier
                    adv_loss = -1.0 * torch.sum(out_tilde_probs * (out_tilde_probs + 1e-9).log())
                else:
                    adv_loss = 0

                # Step 3: Explanation loss:
                exp_loss_s, eloss_list_s = exp_criterion_evaluation(logits_seasonal, beta, exp_criterion)
                exp_loss_t, eloss_list_t = exp_criterion_evaluation(logits_trend, beta, exp_criterion)

                exp_loss = exp_loss_s + exp_loss_t

                # Combine losses:
                loss = clf_loss + exp_loss - gamma * adv_loss

                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

                optimizer.step()

            if adv_flag and ADV:
                cum_adv_opt.append(adv_loss.item())
            else:
                cum_clf_loss.append(clf_loss.item())
                cum_info_loss_T.append(eloss_list_t) 
                cum_info_loss_S.append(eloss_list_s)
                if ADV:
                    cum_adv.append(adv_loss.item())                   
                
                train_loss.append(loss.item())

        if adv_flag and ADV:
            adv_opt_loss = sum(cum_adv_opt) / len(cum_adv_opt)
            print(f'Adv Loss (adv) = {adv_opt_loss:.4f}')
        elif show_mask_sparsity:
            clf_loss = sum(cum_clf_loss) / len(cum_clf_loss)
            info_np_T = np.array(cum_info_loss_T).T
            info_np_S = np.array(cum_info_loss_S).T
            
            info = [sum(c) / len(c) for c in info_np_T]
            info_print_T = ['{:.4f}'.format(l) for l in info]

            info = [sum(c) / len(c) for c in info_np_S]
            info_print_S = ['{:.4f}'.format(l) for l in info]

            if ADV:
                adv_reg_loss = sum(cum_adv) / len(cum_adv)
                print(f'Epoch {epoch}: Info Loss = {info_print} \t Clf Loss = {clf_loss:.4f} \t Adv Loss (reg) = {adv_reg_loss:.4f}')
            else:
                print(f'Epoch {epoch}: Info Loss T = {info_print_T} \t Info Loss S = {info_print_S} \t Clf Loss = {clf_loss:.4f}')

            
        # Validation:
        if ADV:
            adv_predictor.eval()
        extractor.eval()
        predictor.eval()
        with torch.no_grad():
            X, times, y = val_tuple

            #mask, logits, joint_mask = extractor(X, times, captum_input = False)
            masks, logits, joint_masks = extractor(X, times, captum_input = False)
            mask_trend, mask_seasonal = masks
            logits_trend, logits_seasonal = logits
            jm_trend, jm_seasonal = joint_masks
            # print('joint_mask', joint_mask.shape)
            # print('X', X.shape)
            # print('times', times.shape)
            masked_src, masked_times = extractor.apply_masks(X, times, jm_trend, jm_seasonal)
            #masked_src, masked_times, _, _ = extractor.apply_masks(X, times, joint_mask)
            #print('masked_src', masked_src.shape)
            #print('masked_times', masked_times.shape)
            pred = predictor(masked_src, masked_times, captum_input = True)
                
            vloss = clf_criterion(pred, y) #+ exp_criterion_evaluation(mask, beta, exp_criterion)[0] #beta * exp_criterion(mask)

            num_select_T = (mask_trend).sum(dim=1).sum(dim=1).float().mean().item()
            num_select_S = (mask_seasonal).sum(dim=1).sum(dim=1).float().mean().item()
            num_select = num_select_T + num_select_S

            # Calculate F1:
            if performance_scoring == 'f1':
                auc = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')
            elif performance_scoring == 'mae':
                auc = -1.0 * mean_absolute_error(y.cpu().numpy(), pred.cpu().numpy())

            if use_scheduler:
                scheduler.step(-1.0 * vloss) # Step the scheduler based on validation loss

            val_loss.append(vloss)
            val_auc.append(auc)
            val_select.append(num_select)

            if selection_criterion is not None:
                vselect = selection_criterion(auc, num_select / (mask_trend.shape[1] * mask_trend.shape[2]))
            else:
                vselect = -1.0 * vloss if select_by_loss else auc

            cond = (vselect >= min_val_loss)

            if cond: # Want minimum loss or maximum auc
                min_val_loss = vselect
                best_epoch = epoch
                if ADV:
                    torch.save((extractor.state_dict(), predictor.state_dict(), adv_predictor.state_dict()), save_path)
                else:
                    torch.save((extractor.state_dict(), predictor.state_dict()), save_path)

        if (epoch + 1) % 10 == 0: # Print progress:
            print('Epoch {}, Train Loss = {:.4f}, Val F1 = {:.4f}, Num. Selected (T) = {}, Num. Selected (S) = {}'.format(epoch + 1, train_loss[-1], auc, num_select_T, num_select_S))

    # Return best model:
    #model.load_state_dict(torch.load(save_path))
    sdict_tup = torch.load(save_path)
    extractor.load_state_dict(sdict_tup[0])
    predictor.load_state_dict(sdict_tup[1])
    if ADV:
        adv_predictor.load_state_dict(sdict_tup[2])

    if save_path == 'tmp.pt':
        os.remove('tmp.pt') # Remove temporarily stored file

    print('Best AUC achieved at Epoch = {}, AUC = {:.4f}'.format(best_epoch, min_val_loss))

    if ADV:
        return (extractor, predictor, adv_predictor), train_loss, val_loss, val_auc
    return (extractor, predictor), train_loss, val_loss, val_auc