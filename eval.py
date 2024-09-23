from umap import UMAP
import torch

def dim_reduce(embeddings,
               n_components=2,
               n_neighbors=30):
    print("Dimension Reducing...")
    dimred_embeddings = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0, n_jobs=-1).fit_transform(embeddings)
    return dimred_embeddings



def get_masks(labels, original_labels):
    pos_mask = labels[:, 0] == 0
    original_pos_mask = original_labels[:, 0] == 0
    holdout_mask = torch.logical_xor(pos_mask, original_pos_mask)
    neg_mask = torch.logical_and(~pos_mask, ~holdout_mask)

    return pos_mask, holdout_mask, neg_mask


# Function to reduce the amount needed for plotting. 
# Reduces the amount of positives and holdouts for plotting based upon the number of negatives we show
def get_subset_masks(pos_mask, 
                     holdout_mask, 
                     neg_mask, 
                     num_neg=100000): 
    
    scale_factor = num_neg/neg_mask.sum().item()

    pos_remove_ct = int(pos_mask.sum().item() - int(pos_mask.sum().item() * scale_factor))
    holdout_remove_ct = int(holdout_mask.sum().item() - int(holdout_mask.sum().item() * scale_factor))
    neg_remove_ct = int(neg_mask.sum().item() - int(neg_mask.sum().item() * scale_factor))
    
    pos_mask_nonzeros = pos_mask.nonzero().flatten()
    holdout_mask_nonzeros = holdout_mask.nonzero().flatten()
    neg_mask_nonzeros = neg_mask.nonzero().flatten()

    # Reduce number of trues in each mask based upon scale factor. Want total number of trues after scaling to be equal to num_trues * scale_factor
    pos_mask[pos_mask_nonzeros[torch.randperm(pos_mask_nonzeros.size(0))[:pos_remove_ct]]] = False
    holdout_mask[holdout_mask_nonzeros[torch.randperm(holdout_mask_nonzeros.size(0))[:holdout_remove_ct]]] = False
    neg_mask[neg_mask_nonzeros[torch.randperm(neg_mask_nonzeros.size(0))[:neg_remove_ct]]] = False

    return pos_mask, holdout_mask, neg_mask