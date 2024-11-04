from eval import get_masks, dim_reduce, get_subset_masks, plot
import torch
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN, BranchDetector
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score
import const


def cluster_evaluate(embeddings,
                     train_mask,
                     labels, 
                     labels_original,
                     id2rel_original,
                     id2rel_holdout,
                     plot_save_path=None):
    
    pos_mask, holdout_mask, neg_mask = get_masks(labels, labels_original)
    holdout_or_neg_mask = torch.logical_or(holdout_mask, neg_mask) # Holdouts and Negatives
    holdout_cand_mask = torch.logical_and(holdout_mask, ~train_mask)
    assert torch.logical_and(pos_mask, ~train_mask).sum().item() == 0, "Positive examples are outside of the training set, double check sample/mask alignment"

    dimred_embeddings = dim_reduce(embeddings[~train_mask], n_components=10, function='umap', densmap=True) # Use densmap for better local density estimation

    print("Clustering...")
    hdb = HDBSCAN(min_cluster_size=100, min_samples=35, cluster_selection_epsilon=0.5, branch_detection_data=True).fit(dimred_embeddings)
    branch_detector = BranchDetector(min_branch_size=100, label_sides_as_branches=True).fit(hdb) # Since relationships are inherently heirarchical, we can use the branch detector to identify the branches in the cluster tree

    pred_clusters = torch.tensor([-1] * embeddings.size(0)).long()
    pred_clusters[~train_mask] = torch.tensor(branch_detector.labels_).long()

    pseudolabels = torch.tensor([-1] * embeddings.size(0)).long()

    # Noise ratio of holdouts
    holdout_noise_ratio = torch.logical_and(pred_clusters == -1, holdout_cand_mask).sum().item() / holdout_cand_mask.sum().item()

    # Homogenity, Completeness, V-Measure, Adjusted Rand Index applied to the known holdouts as its own individual set (ignoring non-holdout negatives)
    holdout_labels = labels_original[torch.logical_and(pred_clusters != -1, holdout_cand_mask)]
    holdout_label_map = {rel_id.item(): i for i, rel_id in enumerate(torch.unique(labels_original[holdout_mask], dim=0).sum(dim=0).nonzero().flatten())} # Give a number label to each unique relation in the holdout set
    holdout_labels = torch.tensor([holdout_label_map[rel_vec.nonzero().flatten()[0].item()] for rel_vec in holdout_labels]).long()

    holdout_clusters = pred_clusters[torch.logical_and(pred_clusters != -1, holdout_cand_mask)]
    
    holdout_homogeneity, holdout_completeness, holdout_v_measure = homogeneity_completeness_v_measure(holdout_labels, holdout_clusters)
    holdout_ari = adjusted_rand_score(holdout_labels, holdout_clusters)

    holdout_macro_purity = 0
    holdout_macro_completeness = 0
    holdout_max_cluster_count = 0
    all_max_cluster_count = 0

    per_class_stats = {}
    id2rel_holdout_update = {}
    next_pseudolabel = max(list(id2rel_holdout.keys())) + 1
    for label in torch.unique(labels_original[holdout_mask], dim=0):
        for relid in label.nonzero().flatten():
            label_name = id2rel_original[relid.item()]

            if label_name not in per_class_stats: # NOTE: What happens if holdouts share a max cluster??
                label_mask = labels_original[:, relid.item()] == 1

                count = label_mask.sum().item()
                cand_count = torch.logical_and(label_mask, ~train_mask).sum().item()
                clustered_count = torch.logical_and(label_mask, pred_clusters != -1).sum().item()
                noise_ratio = (cand_count - clustered_count) / cand_count if cand_count > 0 else 0

                max_clust = pred_clusters[torch.logical_and(label_mask, pred_clusters != -1)].mode().values.item() # Most common cluster label that is not noise
                id2rel_holdout_update[next_pseudolabel] = label_name
                pseudolabels[pred_clusters == max_clust] = next_pseudolabel
                next_pseudolabel += 1

                max_clust_purity = torch.logical_and(label_mask, pred_clusters == max_clust).sum().item() / (pred_clusters == max_clust).sum().item()
                max_clust_completeness = torch.logical_and(label_mask, pred_clusters == max_clust).sum().item() / cand_count

                holdout_macro_purity += max_clust_purity
                holdout_macro_completeness += max_clust_completeness

                holdout_max_cluster_count += torch.logical_and(label_mask, pred_clusters == max_clust).sum().item()
                all_max_cluster_count += (pred_clusters == max_clust).sum().item()

                per_class_stats[label_name] = {
                    'count': count,
                    'cand_count': cand_count,  
                    'clustered_count': clustered_count,
                    'noise_ratio': noise_ratio,
                    'max_clust': max_clust,
                    'max_clust_purity': max_clust_purity,
                    'max_clust_completeness': max_clust_completeness
                }

    holdout_macro_purity /= len(per_class_stats)
    holdout_macro_completeness /= len(per_class_stats)

    holdout_micro_purity = holdout_max_cluster_count / all_max_cluster_count
    holdout_micro_completeness = holdout_max_cluster_count / holdout_cand_mask.sum().item()

    stats = {
        'original_cts': {
            'sample_ct': len(embeddings), # All samples including those not used for training
            'pos_ct': pos_mask.sum().item(),
            'holdout_ct': holdout_mask.sum().item(),
            'neg_ct': neg_mask.sum().item()
        },
        'train_cts': {
            'train_sample_ct': train_mask.sum().item(),
            'train_pos_ct': pos_mask[train_mask].sum().item(),
            'train_holdout_ct': holdout_mask[train_mask].sum().item(),
            'train_neg_ct': neg_mask[train_mask].sum().item()
        },
        'cand_cts': {
            'cand_sample_ct': (~train_mask).sum().item(),
            'cand_pos_ct': pos_mask[~train_mask].sum().item(),
            'cand_holdout_ct': holdout_mask[~train_mask].sum().item(),
            'cand_neg_ct': neg_mask[~train_mask].sum().item()
        },
        'num_clusters': pred_clusters.unique().size(0),
        'clustered_count': (pred_clusters != -1).sum().item(),
        'holdout_noise_ratio': holdout_noise_ratio,
        'holdout_homogeneity': holdout_homogeneity,
        'holdout_completeness': holdout_completeness,
        'holdout_v_measure': holdout_v_measure,
        'holdout_ari': holdout_ari,
        'holdout_macro_purity': holdout_macro_purity,
        'holdout_macro_completeness': holdout_macro_completeness,
        'holdout_micro_purity': holdout_micro_purity,
        'holdout_micro_completeness': holdout_micro_completeness,
        'per_class_stats': per_class_stats
    }

    if plot_save_path is not None:
        p1_pos_mask, p1_holdout_mask, p1_neg_mask = get_subset_masks(pos_mask, torch.logical_and(holdout_mask, ~train_mask), torch.logical_and(neg_mask, ~train_mask), mode='neg_holdouts')
        plot(embeddings=embeddings,
             labels_original=labels_original,
             id2rel_original=id2rel_original,
             pos_mask=p1_pos_mask,
             holdout_mask=p1_holdout_mask,
             neg_mask=p1_neg_mask,
             cand_mask=(pred_clusters != -1),
             title=f'Embedding Space (Negatives & Holdouts) - {plot_save_path.split("/")[-1]}',
             plot_save_path=plot_save_path,
             cluster_labels=pred_clusters,
             densmap=True)

    return stats, pseudolabels, id2rel_holdout_update