from eval import get_masks
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

def contrastive_evaluate(embeddings, 
                         labels, 
                         labels_original,
                         rel2id_original,
                         id2rel_original,
                         plot_save_path):
    
    pos_mask, holdout_mask, neg_mask = get_masks(labels, labels_original)   

    pos_embeddings = embeddings[pos_mask]
    neg_embeddings = torch.cat([embeddings[neg_mask], embeddings[holdout_mask]], dim=0) # Negatives for evaluation purposes are both negatives and holdouts

    pos_center = pos_embeddings.mean(dim=0)
    neg_center = neg_embeddings.mean(dim=0)

    pos_sims = torch.tensor(cosine_similarity(embeddings, pos_center.unsqueeze(0))).flatten() # Embedding similarities to the positive center

    avg_neg_sim_pos = torch.cat([pos_sims[neg_mask], pos_sims[holdout_mask]]).mean().item() # Average negative similarity to the positive center
    holdout_cand_mask = (pos_sims[holdout_mask] > avg_neg_sim_pos) # Holdouts that are candidates
    neg_cand_mask = (pos_sims[neg_mask] > avg_neg_sim_pos) # Negatives that are candidates

    holdout_cand_ratio = holdout_cand_mask.sum().item() / holdout_mask.sum().item() # Ratio of holdouts that are candidates
    neg_cand_ratio = neg_cand_mask.sum().item() / neg_mask.sum().item() # Ratio of negatives that are candidates
    ratio_all_cand_holdout = holdout_cand_mask.sum().item() / (holdout_cand_mask.sum().item() + neg_cand_mask.sum().item()) # Percentage of all candidates that are holdouts
    ratio_all_cand_neg = neg_cand_mask.sum().item() / (holdout_cand_mask.sum().item() + neg_cand_mask.sum().item()) # Percentage of all candidates that are negatives

    pos_neg_sim = torch.tensor(cosine_similarity(pos_center.unsqueeze(0), neg_center.unsqueeze(0))).flatten().item() # Similarity between the positive and negative centers

    per_class_retain = {}
    holdout_counts = labels_original[holdout_mask].sum(dim=0) # 1D arr of size rel2id_original with counts of each relation in holdouts
    holdout_cand_counts = labels_original[holdout_mask][holdout_cand_mask].sum(dim=0) # 1D arr of size rel2id_original with counts of each relation in holdouts that are candidates
    for label in torch.unique(labels_original[holdout_mask], dim=0):
        for relid in label.nonzero().flatten():
            label_name = id2rel_original[relid.item()]
            count = holdout_counts[relid].item()
            cand_count = holdout_cand_counts[relid].item()
            per_class_retain[label_name] = (cand_count, count) # May overwrite but that's fine

    stats = {
        'pos_neg_sim': pos_neg_sim,
        'holdout_cand_ratio': holdout_cand_ratio,
        'holdout_cand_count': holdout_cand_mask.sum().item(),
        'holdout_count': holdout_mask.sum().item(), 
        'ratio_all_cand_holdout': ratio_all_cand_holdout,
        'neg_cand_ratio': neg_cand_ratio,
        'neg_cand_count': neg_cand_mask.sum().item(), 
        'neg_count': neg_mask.sum().item(),
        'ratio_all_cand_neg': ratio_all_cand_neg,
        'per_class_retain': per_class_retain
    }
    
    return stats

