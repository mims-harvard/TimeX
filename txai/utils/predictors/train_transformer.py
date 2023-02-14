import sys, os
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

sys.path.append(os.path.dirname(__file__))

from .loss import Poly1CrossEntropyLoss

default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 10,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}

def one_hot(y_):
    # Convert y_ to one-hot
    if not (type(y_) is np.ndarray):
        y_ = y_.detach().clone().cpu().numpy() # Assume it's a tensor
    
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def train(
        model, 
        train_loader, 
        val_tuple, # Must be (X, times, y) all in one tensor for each
        n_classes, 
        num_epochs,
        class_weights = None, 
        optimizer = None, 
        standardize = False,
        save_path = None,
        validate_by_step = False,
        criterion = None,
        scheduler_args = default_scheduler_args,
        show_sizes = False,
        regression = False,
        use_scheduler = True,
        counterfactual_training = False,
        max_mask_size = None,
        replace_method = None):
    '''
    Loader should output (B, d, T) - in style of captum input

    Params:
        rand_mask_size (default: None): If an integer is provided, trains model with a 
            random mask generated at test time
        counterfactual_training (bool, optional): If True, counterfactually trains
            the model, as in Hase et al., 2021
        max_mask_size (int, optional): Maximum mask size for counterfactual training 
            procedure
        replace_method (callable, optional): Replacement method to replace values in
            the input when masked out
    '''
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    if criterion is None: # Set if not given
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = Poly1CrossEntropyLoss(
                num_classes = n_classes,
                epsilon = 1.0,
                weight = class_weights,
                reduction = 'mean'
            )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    if save_path is None:
        save_path = 'tmp.pt'

    train_loss, val_auc = [], []
    max_val_auc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        
        # Train:
        model.train()
        for X, times, y in train_loader:

            #print(X.detach().clone().cpu().numpy())
            #print(times.detach().clone().cpu().numpy())

            out = model(X, times, captum_input = True, show_sizes = show_sizes)

            # print('out', out.detach().clone().cpu().numpy())
            # print('y', y.detach().clone().cpu().numpy())

            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if counterfactual_training:
                # calculate loss on replaced values and update
                # Sample 1/2 of batch, run replace:
                batch_size, T, d = X.shape[0], X.shape[1], X.shape[2]
                x_inds = torch.randperm(batch_size)[:(batch_size // 2)]
                xsamp = X[x_inds,:,:]
                masks = torch.ones_like(xsamp).float().to(xsamp.device)
                
                # determine max mask size to sample out (mms)
                mms = max_mask_size if max_mask_size is not None else T * d
                if mms < 1 and mms > 0:
                    mms = X.shape[0] * X.shape[1] * mms # If proportion is provided
                mask_nums = torch.randint(0, high = int(mms), size = ((batch_size // 2),)) 

                # Fill in masks:
                for i in range(masks.shape[0]):
                    cart = torch.cartesian_prod(torch.arange(T), torch.arange(d))[:mask_nums[i]]
                    masks[i,cart[:,0],cart[:,1]] = 0 # Set all spots to 1
                xmasked = replace_method(xsamp, masks)

                out = model(xmasked, times[x_inds,:], captum_input = True, show_sizes = show_sizes)

                optimizer.zero_grad()
                loss2 = criterion(out, y[x_inds])
                loss2.backward()
                optimizer.step()

                loss = loss + loss2 # Add together total loss to be shown in train_loss

            train_loss.append(loss.item())
            
        # Validation:
        model.eval()
        with torch.no_grad():
            X, times, y = val_tuple
            if validate_by_step:
                pred = torch.empty((X.shape[1], n_classes)).to(y.get_device())
                for i in range(X.shape[1]):
                    pred[i,:] = model(X[:,i,:], times[:,i].unsqueeze(-1), show_sizes = show_sizes)
            else:
                pred = model(X, times, show_sizes = show_sizes)
            val_loss = criterion(pred, y)

            # Calculate AUROC:
            #auc = roc_auc_score(one_hot(y), pred.detach().cpu().numpy(), average = 'weighted')
            if regression:
                auc = -1.0 * mean_absolute_error(y.cpu().numpy(), pred.cpu().numpy())
            else:
                auc = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro', )

            if use_scheduler:
                scheduler.step(auc) # Step the scheduler

            val_auc.append(auc)

            if auc > max_val_auc:
                max_val_auc = auc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                #best_sd = model.state_dict()

        if (epoch + 1) % 10 == 0: # Print progress:
            # print('y', y)
            # print('pred', pred) 
            met = 'MAE' if regression else 'F1'
            print('Epoch {}, Train Loss = {:.4f}, Val {} = {:.4f}'.format(epoch + 1, train_loss[-1], met, auc))

    # Return best model:
    model.load_state_dict(torch.load(save_path))

    if save_path == 'tmp.pt':
        os.remove('tmp.pt') # Remove temporarily stored file

    print('Best AUC achieved at Epoch = {}, AUC = {:.4f}'.format(best_epoch, max_val_auc))

    return model, train_loss, val_auc