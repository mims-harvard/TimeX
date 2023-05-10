import torch
import numpy as np

def tune_ptypes(
        model,
        ptype_optimizer,
        train_loader,
        num_epochs,
        sim_criterion,
    ):

    for epoch in range(num_epochs):

        sloss = []

        for X, times, y, ids in train_loader:

            ptype_optimizer.zero_grad()

            out_dict = model(X, times, captum_input = True)
            
            # Just get ptype outs and compare to full outs:
            ptype_z = out_dict['ptypes']
            full_z = out_dict['all_z'][0].detach() # Gradient stoppage

            sim_loss = sim_criterion(ptype_z, full_z)

            sim_loss.backward()
            ptype_optimizer.step()

            sloss.append(sim_loss.detach().clone().item())

        print(f'Epoch: {epoch}: Loss = {np.mean(sloss):.4f}')

    # No validation for now