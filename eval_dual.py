from eval import get_masks, dim_reduce, get_subset_masks, plot
import torch
import matplotlib.pyplot as plt


def dual_evaluate(embeddings,
                  train_mask,
                  labels, 
                  labels_original,
                  id2rel_original,
                  plot_save_path=None):
    
    pos_mask, holdout_mask, neg_mask = get_masks(labels, labels_original)
    holdout_or_neg_mask = torch.logical_or(holdout_mask, neg_mask) # Holdouts and Negatives
    assert torch.logical_and(pos_mask, ~train_mask).sum().item() == 0, "Positive examples are outside of the training set, double check sample/mask alignment"

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
        }
    }

    if plot_save_path is not None:
        p1_pos_mask, p1_holdout_cand_mask, p1_neg_cand_mask = get_subset_masks(pos_mask, torch.logical_and(holdout_mask, ~train_mask), torch.logical_and(neg_mask, ~train_mask), mode='only_holdout') # pos mask will be empty, NOTE: only plotting candidates
        plot(embeddings=embeddings,
             labels_original=labels_original,
             id2rel_original=id2rel_original,
             pos_mask=p1_pos_mask,
             holdout_mask=p1_holdout_cand_mask,
             neg_mask=p1_neg_cand_mask,
             cand_mask=torch.ones_like(holdout_mask).bool(),
             title=f'Dual-Objective Embedding Space (Only Candidates) - {plot_save_path.split("/")[-1]}',
             plot_save_path=plot_save_path.replace('.png', '_candidates_no-filter.png'))
        
        p2_pos_mask, p2_holdout_cand_mask, p2_neg_cand_mask = get_subset_masks(pos_mask, torch.logical_and(holdout_mask, ~train_mask), torch.logical_and(neg_mask, ~train_mask), mode='all_pos') # pos mask will be empty, NOTE: only plotting candidates
        plot(embeddings=embeddings,
             labels_original=labels_original,
             id2rel_original=id2rel_original,
             pos_mask=p2_pos_mask,
             holdout_mask=p2_holdout_cand_mask,
             neg_mask=p2_neg_cand_mask,
             cand_mask=torch.ones_like(holdout_mask).bool(),
             title=f'Dual-Objective Embedding Space (Positives & Candidates) - {plot_save_path.split("/")[-1]}',
             plot_save_path=plot_save_path.replace('.png', '_pos-and-cand_no-filter.png'))

    return stats