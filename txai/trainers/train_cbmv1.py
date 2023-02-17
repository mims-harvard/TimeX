import torch
import numpy as np
import ipdb

from txai.utils.predictors.loss import exp_criterion_evaluation
from txai.utils.predictors.eval import eval_cbmv1

def train_cbmv1(
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

    try:
        model.mu, model.sigma_inv
    except:
        raise ValueError('Must store concept bank before running the trainer')

    best_epoch = 0
    best_val_metric = -1e9

    for epoch in range(num_epochs):
        
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss = [], [], []
        for X, times, y in train_loader:

            optimizer.zero_grad()

            #ipdb.set_trace()

            out, concept_scores, masks, logits = model(X, times, captum_input = True)

            if out.isnan().sum() > 0:
                print('out', out.isnan().sum())
                exit()

            clf_loss = clf_criterion(out, y)

            total_eloss_list = []
            for i in range(len(logits)):
                if i == 0:
                    # Multiplies beta to a list of exp criterions or just multiplies if beta is constant and exp_criterion is direct
                    exp_loss, eloss_list = exp_criterion_evaluation(logits[i], beta, exp_criterion)
                else:
                    exp_loss_i, eloss_list = exp_criterion_evaluation(logits[i], beta, exp_criterion)
                    exp_loss += exp_loss_i
                total_eloss_list.append(eloss_list)

            exp_loss /= len(logits) # Normalize out with respect to number of masks in model

            loss = clf_loss + exp_loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            cum_sparse.append([(mask.sum() / mask.flatten().shape[0]).item() for mask in masks])
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append(total_eloss_list)

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean(axis=0)
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()

        print(f'Epoch: {epoch}: Sparsity = {sparse} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f}')
        #exit()

        # Eval after every epoch
        # Call evaluation function:
        f1, (_, concept_scores, masks, logits) = eval_cbmv1(val_tuple, model)

        # Early stopping procedure:
        if f1 > best_val_metric:
            best_val_metric = f1
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch


        if (epoch + 1) % 10 == 0:
            valsparse = ['{:.4f}'.format(masks[i].mean().item()) for i in range(len(masks))]
            concept_avg = concept_scores.mean(dim=0)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}, Concept Avg = {concept_avg.detach().cpu().numpy()}')


    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')

    model.load_state_dict(torch.load(save_path))

    return model
