import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import drop_feature


class contrast_generator(nn.Module):
    def __init__(self, predict_model):
        super(contrast_generator, self).__init__()
        self.predict_model = predict_model
        self.device = 'cuda'
        self.drop_prob = 0.5

    def forward(self, model, X, pos_num, neg_num, times):
        pos_mask = torch.tensor(np.random.choice([1, 0], size=(X.shape)), device=self.device)
        pos_mask = torch.cuda.FloatTensor(X.shape).uniform_() > 0.8
        pos_tensor = X.mul(pos_mask)
        _, tar_exp,_ = model(X,times, get_agg_embed=True)
        _, pos_exp,_ = model(pos_tensor,times, get_agg_embed=True)
        return tar_exp, pos_exp

    
    def pos_drop_sampling_mask(self, data, S):
        with torch.no_grad():
            sample_num = S.shape[1]
            pos_tensor = data.mul(S)
            ref_score  = self.predict_model.predict(data)
            pred_score = self.predict_model.predict(pos_tensor)
            try:
                pos_gap = np.concatenate((pos_gap, pred_score), axis=1)
            except:
                pos_gap = pred_score
            score_gap = np.absolute(pos_gap - ref_score)
            rank_pos_list = np.argmin(score_gap, axis=1)
            best_pos = np.stack([ x[rank_pos_list[idx]] for idx, x in enumerate(train) ])
        return best_pos