import math
import torch
import torch.nn.functional as F

from txai.utils.functional import js_divergence, js_divergence_logsoftmax

class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature = 1.0):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, positives, negatives, get_all_scores = False):    
        '''
        embeddings: (B, d) shape
        positives: (B, d, n_pos) shape
        negatives: (B, d, n_neg) shape
        '''
        
        embeddings = F.normalize(embeddings.unsqueeze(1), dim = -1) # Add 1 to embeddings dimension

        # print('embeddings', embeddings.shape)
        # print('pos', positives.shape)
        # print('negatives', negatives.shape)

        # Sim to positives:
        sim_pos = torch.bmm(embeddings, F.normalize(positives, dim = 1).unsqueeze(-1)) / self.temperature
        sim_neg = torch.bmm(embeddings, F.normalize(negatives, dim = 1)) / self.temperature

        # print('pos score', sim_pos.exp().sum(dim=-1))
        # print('neg score', sim_neg.exp().sum(dim=-1))

        score = -1.0 * torch.log(sim_pos.exp().sum(dim=-1) / sim_neg.exp().sum(dim=-1))

        if get_all_scores:
            return score.mean(), sim_pos.exp().sum(dim=-1), sim_neg.exp().sum(dim=-1)
        else:
            return score.mean()

class LabelConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(LabelConsistencyLoss, self).__init__()

    def forward(self, mask_labels, full_labels):    
        '''
        embeddings: (B, d) shape
        positives: (B, d, n_pos) shape
        negatives: (B, d, n_neg) shape
        '''
        
        #embeddings = F.normalize(embeddings.unsqueeze(1), dim = -1) # Add 1 to embeddings dimension
        # mask_labels = mask_labels.softmax(dim=-1)
        # full_labels = full_labels.softmax(dim=-1)
        mask_labels_ls = mask_labels.log_softmax(dim=-1)
        full_labels_ls = full_labels.log_softmax(dim=-1)

        # if torch.any(torch.isnan(mask_labels_ls)):
        #     print("NAN mask_labels")
        #     import ipdb; ipdb.set_trace()
        # assert not torch.any(torch.isnan(full_labels_ls)), "NAN full_labels"


        # Enumerate batch:
        combs = torch.combinations(torch.arange(mask_labels_ls.shape[0]), r = 2, with_replacement = False)

        mask_labels_expanded_lhs = mask_labels_ls[combs[:,0],:]
        mask_labels_expanded_rhs = mask_labels_ls[combs[:,1],:]

        full_labels_expanded_lhs = full_labels_ls[combs[:,0],:]    
        full_labels_expanded_rhs = full_labels_ls[combs[:,1],:]

        # print('embeddings', embeddings.shape)
        # print('pos', positives.shape)
        # print('negatives', negatives.shape)

        # Sim to positives:
        score_mask = js_divergence(mask_labels_expanded_lhs, mask_labels_expanded_rhs, log_already = True)
        score_full = js_divergence(full_labels_expanded_lhs, full_labels_expanded_rhs, log_already = True)

        score = (score_mask - score_full).pow(2).mean()
        #score = (score_mask -  score_full).abs().mean()

        #import ipdb; ipdb.set_trace()

        # print('pos score', sim_pos.exp().sum(dim=-1))
        # print('neg score', sim_neg.exp().sum(dim=-1))

        #score = -1.0 * torch.log(sim_pos.exp().sum(dim=-1) / sim_neg.exp().sum(dim=-1))

        # if get_all_scores:
        #     return score.mean(), sim_pos.exp().sum(dim=-1), sim_neg.exp().sum(dim=-1)
        # else:
        return score

class LabelConsistencyLoss_Forecasting(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mask_pred, full_pred):
        # Expand via combs, same as in original LabelConsistency loss
        combs = torch.combinations(torch.arange(mask_pred.shape[0]), r = 2, with_replacement = False)

        mask_pred_expanded_lhs = mask_pred[combs[:,0],:]
        mask_pred_expanded_rhs = mask_pred[combs[:,1],:]

        full_pred_expanded_lhs = full_pred[combs[:,0],:]    
        full_pred_expanded_rhs = full_pred[combs[:,1],:]

        mask_pred_score = (mask_pred_expanded_lhs - mask_pred_expanded_rhs).pow(2).mean(dim=1)
        full_pred_score = (full_pred_expanded_lhs - full_pred_expanded_rhs).pow(2).mean(dim=1)

        score = (mask_pred_score - full_pred_score).pow(2).mean()

        return score

class LabelConsistencyLoss_LS(torch.nn.Module):
    def __init__(self):
        super(LabelConsistencyLoss_LS, self).__init__()

    def forward(self, mask_labels, full_labels):    
        '''
        embeddings: (B, d) shape
        positives: (B, d, n_pos) shape
        negatives: (B, d, n_neg) shape
        '''

        # Enumerate batch:
        combs = torch.combinations(torch.arange(mask_labels.shape[0]), r = 2, with_replacement = False)

        mask_labels_expanded_lhs = mask_labels[combs[:,0],:]
        mask_labels_expanded_rhs = mask_labels[combs[:,1],:]

        full_labels_expanded_lhs = full_labels[combs[:,0],:]    
        full_labels_expanded_rhs = full_labels[combs[:,1],:]

        # Sim to positives:
        score_mask = js_divergence_logsoftmax(mask_labels_expanded_lhs, mask_labels_expanded_rhs)
        score_full = js_divergence_logsoftmax(full_labels_expanded_lhs, full_labels_expanded_rhs)

        score = (score_mask - score_full).pow(2).mean()

        return score

class LabelAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(LabelAlignmentLoss, self).__init__()

    def forward(self, mask_labels, full_labels):
        mask_p = mask_labels.softmax(dim=-1)
        full_p = full_labels.softmax(dim=-1)
        return js_divergence(mask_p, full_p)

# class LabelAlignmentLoss_LS(torch.nn.Module):
#     def __init__(self):
#         super(LabelAlignmentLoss_LS, self).__init__()

#     def forward(self, mask_labels, full_labels):
#         return js_divergence_logsoftmax(mask_labels, full_labels)

class ConceptTopologyLoss(torch.nn.Module):
    def __init__(self, temperature = 1.0, prop_select = 0.5):
        super(ConceptTopologyLoss, self).__init__()
        self.temperature = temperature
        self.prop_select = prop_select
    
    def forward(self, original_embeddings, concept_embeddings):

        assert (original_embeddings.shape == concept_embeddings.shape)

        # Normalize embeddings and get similarities of all (outer product to get all pairs)
        original_embeddings = F.normalize(original_embeddings, dim = -1)
        original_sim_mat = torch.matmul(original_embeddings, original_embeddings.transpose(0,1))
        original_sim_mat = original_sim_mat.flatten()

        concept_embeddings = F.normalize(concept_embeddings, dim = -1)
        concept_sim_mat = torch.matmul(concept_embeddings, concept_embeddings.transpose(0,1)).flatten()

        # print('o', torch.any(torch.isnan(original_sim_mat)))
        # print('c', torch.any(torch.isnan(concept_sim_mat)))

        n_sample = math.floor(self.prop_select * original_sim_mat.shape[0]) 

        rand_sample = torch.randperm(concept_sim_mat.shape[0])[:n_sample]
        
        scores = (original_sim_mat[rand_sample] - concept_sim_mat[rand_sample]).abs()

        if torch.any(torch.isnan(scores)):
            raise ValueError('ALERT - ConceptTopologyLoss has nan')

        return scores.mean()

class EmbedConsistencyLoss(torch.nn.Module):
    def __init__(self, normalize_distance = False):
        super(EmbedConsistencyLoss, self).__init__()
        self.normalize_distance = normalize_distance

    def forward(self, original_embeddings, concept_embeddings):
        original_embeddings = F.normalize(original_embeddings, dim = -1)
        concept_embeddings = F.normalize(concept_embeddings, dim = -1)

        original_sim_mat = torch.matmul(original_embeddings, original_embeddings.transpose(0,1)) # Size (B, B)
        concept_sim_mat = torch.matmul(concept_embeddings, concept_embeddings.transpose(0,1)) # Size (B, B)

        # print('original_sim_mat', original_sim_mat.mean())
        # print('concept_sim_mat', concept_sim_mat.mean())

        # Normalize by batch:
        # if self.normalize_distance:
        #     original_sim_mat = original_sim_mat / original_sim_mat.std()
        #     concept_sim_mat = concept_sim_mat / concept_sim_mat.std()

        score = (original_sim_mat - concept_sim_mat).pow(2).mean()

        if self.normalize_distance:
            score /= original_sim_mat.var()

        #import ipdb; ipdb.set_trace()

        return score 

class SimCLRwConsistencyLoss(torch.nn.Module):
    def __init__(self, lam = 1.0, temperature = 1.0):
        super(SimCLRwConsistencyLoss, self).__init__()
        self.lam = lam
        self.simclr_loss = SimCLRLoss(temperature = temperature)
        self.con_loss = EmbedConsistencyLoss()
    def forward(self, embeddings, positives, negatives):
        sclr = self.simclr_loss(embeddings, positives, negatives)
        con = self.con_loss(positives, embeddings)

        return sclr + self.lam * con


class GeneralScoreContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature = 1.0):
        super(GeneralScoreContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores, get_all_scores = False):
        pos_scores = (pos_scores.squeeze() / self.temperature)
        neg_scores = (neg_scores.squeeze() / self.temperature)
        score = -1.0 * torch.log(pos_scores.exp().sum(dim=-1) / neg_scores.exp().sum(dim=-1))

        if get_all_scores:
            return score.mean(), pos_scores.exp().sum(dim=-1), neg_scores.exp().sum(dim=-1)
        else:
            return score.mean()
