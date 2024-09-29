from umap import UMAP
import torch

def dim_reduce(embeddings,
               n_components=2,
               n_neighbors=30):
    print("Dimension Reducing...")
    dimred_embeddings = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0, n_jobs=-1).fit_transform(embeddings)
    return dimred_embeddings



def get_masks(labels, labels_original):
    pos_mask = labels[:, 0] == 0
    original_pos_mask = labels_original[:, 0] == 0
    holdout_mask = torch.logical_xor(pos_mask, original_pos_mask)
    neg_mask = torch.logical_and(~pos_mask, ~holdout_mask)

    return pos_mask, holdout_mask, neg_mask



# Function to reduce the amount needed for plotting. 
# all_pos will retain all positives and holdouts but will fill the rest with negatives up to num total samples
def get_subset_masks(pos_mask, 
                     holdout_mask, 
                     neg_mask, 
                     num=75000,
                     mode='scale'): # Mode can be 'scale' or 'all_pos'
    if mode == 'scale':
        subset_pos_mask = torch.zeros(pos_mask.size(0)).bool()
        subset_holdout_mask = torch.zeros(holdout_mask.size(0)).bool()
        subset_neg_mask = torch.zeros(neg_mask.size(0)).bool()

        scale_factor = num/(pos_mask.sum().item() + holdout_mask.sum().item() + neg_mask.sum().item())

        pos_mask_nonzeros = pos_mask.nonzero().flatten()
        holdout_mask_nonzeros = holdout_mask.nonzero().flatten()
        neg_mask_nonzeros = neg_mask.nonzero().flatten()

        num_pos = int(pos_mask.sum().item() * scale_factor)
        num_holdout = int(holdout_mask.sum().item() * scale_factor)
        num_neg = int(neg_mask.sum().item() * scale_factor)

        subset_pos_mask[pos_mask_nonzeros[torch.randperm(pos_mask_nonzeros.size(0))[:num_pos]]] = True
        subset_holdout_mask[holdout_mask_nonzeros[torch.randperm(holdout_mask_nonzeros.size(0))[:num_holdout]]] = True
        subset_neg_mask[neg_mask_nonzeros[torch.randperm(neg_mask_nonzeros.size(0))[:num_neg]]] = True

    elif mode == 'all_pos':
        subset_pos_mask = pos_mask
        subset_holdout_mask = holdout_mask
        subset_neg_mask = torch.zeros(neg_mask.size(0)).bool()

        num_negs = num - pos_mask.sum().item() - holdout_mask.sum().item()
        if num_negs > 0:
            neg_mask_nonzeros = neg_mask.nonzero().flatten()
            subset_neg_mask[neg_mask_nonzeros[torch.randperm(neg_mask_nonzeros.size(0))[:num_negs]]] = True

    return subset_pos_mask, subset_holdout_mask, subset_neg_mask