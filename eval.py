from umap import UMAP
import torch

def dim_reduce(embeddings,
               n_components=2,
               n_neighbors=30):
    dimred_embeddings = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0, n_jobs=-1).fit_transform(embeddings)
    return dimred_embeddings



def get_masks(labels, original_labels):
    pos_mask = labels[:, 0] == 0
    original_pos_mask = original_labels[:, 0] == 0
    holdout_mask = torch.logical_xor(pos_mask, original_pos_mask)
    neg_mask = torch.logical_and(~pos_mask, ~holdout_mask)

    return pos_mask, holdout_mask, neg_mask