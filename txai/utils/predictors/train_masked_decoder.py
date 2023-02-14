import sys, os
import torch
import numpy as np
from tqdm import trange

from .train_transformer import default_scheduler_args
from txai.utils.masking import random_time_mask

from sklearn.metrics import roc_auc_score, f1_score
from loss import Poly1CrossEntropyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_masked_encdec(
        model, 
        train_loader, 
        val_tuple, # Must be (X, times, y) all in one tensor for each
        n_classes, 
        num_epochs,
        clf_criterion,
        exp_criterion,
        recon_criterion,
        delay_phi = 0,
        train_random = None,
        use_zero_vec_perturb = False,
        beta = 1,
        gamma = 1,
        optimizer = None, 
        standardize = False,
        save_path = None,
        validate_by_step = False,
        show_mask_sparsity = False,
        scheduler_args = default_scheduler_args,
        use_scheduler = True,
        clip_norm = False,
        select_by_loss = True,
        gamma_decay_beta = None,
        use_neg_mask_loss = False,
        extractor_loss_lower_bound = 1e-4,
        add_beta_factor = 5,
        optimize_separate = False,
        use_full_samples = False,
        selection_criterion = None):
    '''
    Loader should output (B, d, T) - in style of captum input

    NOTE: Model needs to not return original attention (i.e. return_org_attn = False)

    clf_criterion: Loss function for classification
        - Must have forward signature (logits, labels)
    exp_criterion: Loss function for explainer
        - Must have forward signature (attention)
    delay_phi: Number of epochs for which to train prediction encoder before training explainer
    train_random (bool, optional): If provided, use random masking on the inputs when 
        pre-training delay_phi.
    '''
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 0.01)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_args
        )

    if save_path is None:
        save_path = 'tmp.pt'

    pretrain_train_loss, pretrain_val_f1 = [], []
    for epoch in trange(delay_phi):
        # Train only classifier

        model.train()

        for X, times, y in train_loader:

            out = model.enc_theta(X, times, None, captum_input = True)

            optimizer.zero_grad()
            loss = clf_criterion(out, y)

            if train_random is not None:
                # Choose some random number of samples in the batch:
                num_perturb = torch.randint(1,X.shape[0], size = (1,))[0].item() # Number of samples to perturb
                to_perturb = torch.randperm(X.shape[0])[:num_perturb] # Which samples should be perturbed

                # Makes filler vector, i.e. one that receives
                if use_zero_vec_perturb:
                    tofill = torch.zeros(X.shape[1], X.shape[2]).to(device)
                else:
                    torch.manual_seed(1234) # FIXED, can change later
                    tofill = torch.randn(X.shape[1]).unsqueeze(-1).repeat(1,X.shape[2]).to(device)

                # Make random mask based on train_random rate
                msize = (X.shape[1], X.shape[2])
                masked_samples, masked_times = [], []
                ymask = torch.zeros(num_perturb).long().to(device)
                i = 0
                for t in to_perturb:
                    mask = random_time_mask(rate = (1 - train_random), size = msize).to(device)

                    new_sample = (mask) * X[t] + (1 - mask) * tofill
                    masked_samples.append(new_sample)
                    masked_times.append(times[t])
                    ymask[i] = y[t]
                    i += 1

                Xmask = torch.stack(masked_samples).to(device)
                Tmask = torch.stack(masked_times).to(device)

                outmask = model.enc_theta(Xmask, Tmask, None, captum_input = True)
                loss_mask = clf_criterion(outmask, ymask)
                loss += loss_mask
            
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # Use no mask on the model output
            X, times, y = val_tuple
            if validate_by_step:
                pred = torch.empty((X.shape[1], n_classes)).to(y.get_device())
                for i in range(X.shape[1]):
                    pred[i,:] = model.enc_theta(X[:,i,:], times[:,i].unsqueeze(-1), None, captum_input = False)
            else:
                pred = model.enc_theta(X, times, None, captum_input = False)

            # Calculate F1:
            f1 = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

            # if use_scheduler:
            #     scheduler.step(f1) # Step the scheduler

            pretrain_val_f1.append(f1)

        # Don't use scheduler on delay_phi

        if ((epoch + 1) % 10 == 0) or (epoch == (delay_phi - 1)): # Print progress:
            print('Pretrain Epoch {}, Train Loss = {:.4f}, Val F1 = {:.4f}'.format(epoch + 1, loss.item(), f1))

    train_loss, val_loss, val_auc, val_select = [], [], [], []
    best_epoch = 0
    min_val_loss = -1e10
    for epoch in range(num_epochs):
        
        # Decays beta if needed:
        if gamma_decay_beta:
            #beta = beta * np.exp(-1.0 * gamma_decay_beta * epoch)
            beta += (add_beta_factor/num_epochs) * beta 

        # Train:
        model.train()
        cum_sparse, cum_info_loss, cum_clf_loss, cum_recon_loss = [], [], [], []
        for X, times, y in train_loader:

            optimizer.zero_grad()

            if optimize_separate:
                out, mask, to_opt, decoded = model(X, times, captum_input = True)
                exp_loss = exp_criterion(to_opt)
            else:
                out, mask, decoded = model(X, times, captum_input = True)
                exp_loss = exp_criterion(mask)

            # Get combination loss
            clf_loss = clf_criterion(out, y)
            rec_loss = recon_criterion(decoded, X)

            if use_neg_mask_loss:
                # Invert given mask
                # print('mask size', mask.shape)
                # print('src size', X.shape)
                Xpert, timepert = model.apply_mask(X, times, mask = (1 - mask), captum_input = True)
                outpert = model.enc_theta(Xpert, timepert, captum_input = True)
                clf_neg_loss = -1.0 * clf_criterion(outpert, y)
                clf_loss = 10 * clf_loss + clf_neg_loss

            if use_full_samples:
                out = model.enc_theta(X, times, captum_input = True)
                clf_loss2 = clf_criterion(out, y)
                clf_loss += clf_loss2

            # Total loss computation:
            loss = clf_loss + beta * exp_loss + gamma * rec_loss

            if exp_loss.item() < extractor_loss_lower_bound:
                beta -= (add_beta_factor/num_epochs) * beta

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            if show_mask_sparsity:
                cum_sparse.append(((mask).sum() / mask.flatten().shape[0]).item())
                cum_clf_loss.append(clf_loss.item())
                cum_info_loss.append(exp_loss.item())
                cum_recon_loss.append(rec_loss.item())
                
            train_loss.append(loss.item())

        if show_mask_sparsity:
            sparse = sum(cum_sparse) / len(cum_sparse)
            clf_loss = sum(cum_clf_loss) / len(cum_clf_loss)
            info = sum(cum_info_loss) / len(cum_info_loss)
            rec = sum(cum_recon_loss) / len(cum_recon_loss)
            print(f'Epoch {epoch}: Sparsity = {sparse:.10f} \t Info Loss = {info:.4f} \t Clf Loss = {clf_loss:.4f} \t Recon Loss = {rec:.4f}')

            
        # Validation:
        model.eval()
        with torch.no_grad():
            X, times, y = val_tuple
            if validate_by_step:
                pred = torch.empty((X.shape[1], n_classes)).to(y.get_device())
                mask = torch.empty((X.shape[1],X.shape[0])).to(y.get_device())
                for i in range(X.shape[1]):
                    pred[i,:], mask[i,:]  = model(X[:,i,:], times[:,i].unsqueeze(-1))
                    #pred[i,:], sat_mask[i,:], _ = model(X[:,i,:], times[:,i].unsqueeze(-1))
            else:
                pred, mask = model(X, times, captum_input = False)
                
            #_, val_loss_dict = criterion(sat_mask, pred, y) 
            vloss = clf_criterion(pred, y) + beta * exp_criterion(mask)

            num_select = (mask).sum(dim=1).sum(dim=1).float().mean().item()

            # Calculate F1:
            #auc = roc_auc_score(one_hot(y), pred.detach().cpu().numpy(), average = 'weighted')
            auc = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')

            if use_scheduler:
                scheduler.step(-1.0 * vloss) # Step the scheduler based on validation loss

            val_loss.append(vloss)
            val_auc.append(auc)
            val_select.append(num_select)

            if selection_criterion is not None:
                vselect = selection_criterion(auc, num_select / (mask.shape[1] * mask.shape[2]))
            else:
                vselect = -1.0 * vloss if select_by_loss else auc

            cond = (vselect >= min_val_loss)

            if cond: # Want minimum loss or maximum auc
                min_val_loss = vselect
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                #best_sd = model.state_dict()

        if (epoch + 1) % 10 == 0: # Print progress:
            print('Epoch {}, Train Loss = {:.4f}, Val F1 = {:.4f}, Num. Selected = {}'.format(epoch + 1, train_loss[-1], auc, num_select))

    # Return best model:
    model.load_state_dict(torch.load(save_path))

    if save_path == 'tmp.pt':
        os.remove('tmp.pt') # Remove temporarily stored file

    print('Best AUC achieved at Epoch = {}, AUC = {:.4f}'.format(best_epoch, min_val_loss))

    return model, train_loss, val_loss, val_auc