from eval import get_masks, get_subset_masks, dim_reduce, plot
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def contrastive_evaluate(embeddings, 
                         labels, 
                         labels_original,
                         id2rel_original,
                         plot_save_path=None):

    pos_mask, holdout_mask, neg_mask = get_masks(labels, labels_original)
    holdout_or_neg_mask = torch.logical_or(holdout_mask, neg_mask) # Holdouts and Negatives

    pos_center = embeddings[pos_mask].mean(dim=0)
    neg_center = embeddings[holdout_or_neg_mask].mean(dim=0)

    pca_embeddings = dim_reduce(embeddings, n_components=96, function='pca') # Can mess with dimension reduction and mean_init initialization, 64 dim with pos/neg centers seems good. Semi-supervised gaussian mixture?
    pos_embeddings = pca_embeddings[pos_mask]
    neg_embeddings = pca_embeddings[holdout_or_neg_mask] # Negatives for evaluation purposes are both negatives and holdouts

    gmm = GaussianMixture(n_components=2, verbose=1, means_init=torch.stack([pos_embeddings.mean(dim=0), neg_embeddings.mean(dim=0)], dim=0), verbose_interval=1) # initialize with means?
    gmm.fit(torch.cat([pos_embeddings, neg_embeddings[torch.randperm(neg_embeddings.shape[0])][:]], dim=0)) # can fit to subsample of negatives 400,000 was good
    
    pred_clusters = torch.tensor(gmm.predict(pca_embeddings)) # Predict clusters based upon the embeddings
    pos_cluster_label = pred_clusters[pos_mask].mode().values.item()

    cand_mask = torch.logical_and((pred_clusters == pos_cluster_label), holdout_or_neg_mask) # Candidates Mask for samples that will be used for inference/validation in the dual-objective model
    holdout_cand_mask = torch.logical_and(holdout_mask, cand_mask) # Holdouts that are candidates
    neg_cand_mask = torch.logical_and(neg_mask, cand_mask) # TRUE negatives that are candidates

    per_class_retain = {}
    holdout_counts = labels_original[holdout_mask].sum(dim=0) # 1D arr of size rel2id_original with counts of each relation in holdouts
    holdout_cand_counts = labels_original[holdout_cand_mask].sum(dim=0) # 1D arr of size rel2id_original with counts of each relation in holdouts that are candidates
    for label in torch.unique(labels_original[holdout_mask], dim=0):
        for relid in label.nonzero().flatten():
            label_name = id2rel_original[relid.item()]
            count = holdout_counts[relid].item()
            cand_count = holdout_cand_counts[relid].item()
            per_class_retain[label_name] = (cand_count, count, cand_count/count) # May overwrite but that's fine

    stats = {
        'pos_neg_sim': torch.tensor(cosine_similarity(pos_center.unsqueeze(0), neg_center.unsqueeze(0))).flatten().item(), # Similarity between the positive and negative centers
        'sample_ct': len(embeddings), # All samples including those not used for training
        'cand_ct': cand_mask.sum().item(),
        'positives': {'pos_ct': pos_mask.sum().item()},
        'holdouts': {'holdout_ct': holdout_mask.sum().item(),  # Number of holdouts in dataset
                     'holdout_cand_ct': holdout_cand_mask.sum().item(), # Number of candidate holdouts based upon current training set
                     'holdout_cand_ratio': holdout_cand_mask.sum().item() / holdout_mask.sum().item(), # Ratio of holdouts that are candidates
                     'ratio_all_cand_holdout': holdout_cand_mask.sum().item() / cand_mask.sum().item()}, # Ratio of all candidates that are holdouts
        'negatives': {'neg_ct': neg_mask.sum().item(),  # Number of negatives in dataset
                      'neg_cand_ct': neg_cand_mask.sum().item(), # Number of candidate holdouts based upon current training set
                      'neg_cand_ratio': neg_cand_mask.sum().item() / neg_mask.sum().item(), # Ratio of negatives that are candidates
                      'ratio_all_cand_neg': neg_cand_mask.sum().item() / cand_mask.sum().item()}, # Ratio of all candidates that are negatives
        'per_class_retain': per_class_retain
    }


    if plot_save_path is not None:
        p1_pos_mask, p1_holdout_mask, p1_neg_mask = get_subset_masks(pos_mask, holdout_mask, neg_mask, mode='all_pos')
        plot(embeddings=embeddings,
             labels_original=labels_original,
             id2rel_original=id2rel_original,
             pos_mask=p1_pos_mask,
             holdout_mask=p1_holdout_mask,
             neg_mask=p1_neg_mask,
             cand_mask=cand_mask,
             title=f'Contrastive Embedding Space (All Positives & Holdouts) - {plot_save_path.split("/")[-1]}',
             plot_save_path=plot_save_path.replace('.png', '_all-pos.png'))

        p2_pos_mask, p2_holdout_mask, p2_neg_mask = get_subset_masks(pos_mask, holdout_mask, neg_mask, mode='scale')
        plot(embeddings=embeddings,
             labels_original=labels_original,
             id2rel_original=id2rel_original,
             pos_mask=p2_pos_mask,
             holdout_mask=p2_holdout_mask,
             neg_mask=p2_neg_mask,
             cand_mask=cand_mask,
             title=f'Contrastive Embedding Space (Scaled All) - {plot_save_path.split("/")[-1]}',
             plot_save_path=plot_save_path.replace('.png', '_scale.png'))
    
    return stats, cand_mask
    

