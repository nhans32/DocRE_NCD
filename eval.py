from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

def dim_reduce(embeddings,
               n_components=2,
               n_neighbors=30,
               function='umap'):
    print("Dimension Reducing...")
    if function == 'umap':
        dimred = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0, n_jobs=-1)
    elif function == 'tsne':
        dimred = TSNE(n_components=n_components, perplexity=n_neighbors, n_jobs=-1)
    elif function == 'pca':
        dimred = PCA(n_components=n_components)
    dimred_embeddings = torch.tensor(dimred.fit_transform(embeddings))
    return dimred_embeddings



def get_masks(labels, labels_original):
    pos_mask = labels[:, 0] == 0
    original_pos_mask = labels_original[:, 0] == 0
    holdout_mask = torch.logical_xor(pos_mask, original_pos_mask)
    neg_mask = torch.logical_and(~pos_mask, ~holdout_mask)

    assert torch.logical_xor(pos_mask, torch.logical_xor(neg_mask, holdout_mask)).all(), "Masks are not mutually exclusive and exhaustive" # Check that the masks are mutually exclusive and exhaustive

    return pos_mask, holdout_mask, neg_mask



# Function to reduce the amount needed for plotting. 
# all_pos will retain all positives and holdouts but will fill the rest with negatives up to num total samples
def get_subset_masks(pos_mask, 
                     holdout_mask, 
                     neg_mask, 
                     num=75000,
                     mode='scale'): # Mode can be 'scale', 'all_pos' or 'only_holdout'
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

    elif mode == 'only_holdout':
        subset_pos_mask = torch.zeros(pos_mask.size(0)).bool()
        subset_holdout_mask = holdout_mask
        subset_neg_mask = torch.zeros(neg_mask.size(0)).bool()

        num_negs = num - holdout_mask.sum().item()
        if num_negs > 0:
            neg_mask_nonzeros = neg_mask.nonzero().flatten()
            subset_neg_mask[neg_mask_nonzeros[torch.randperm(neg_mask_nonzeros.size(0))[:num_negs]]] = True

    return subset_pos_mask, subset_holdout_mask, subset_neg_mask



def plot(embeddings,
         labels_original,
         id2rel_original,
         pos_mask,
         holdout_mask,
         neg_mask,
         cand_mask,
         title,
         plot_save_path,
         reduce_fn='umap'):

    pos_mean = embeddings[neg_mask][0].unsqueeze(0) if pos_mask.sum().item() == 0 else embeddings[pos_mask].mean(dim=0).unsqueeze(0) # have placeholder
    neg_mean = embeddings[pos_mask][0].unsqueeze(0) if neg_mask.sum().item() == 0 else embeddings[torch.logical_or(holdout_mask, neg_mask)].mean(dim=0).unsqueeze(0)

    subset_embeddings = torch.cat([embeddings[pos_mask], embeddings[holdout_mask], embeddings[neg_mask], pos_mean, neg_mean], dim=0)

    subset_pos_mask = torch.cat([torch.ones(pos_mask.sum().item()), torch.zeros(holdout_mask.sum().item() + neg_mask.sum().item()), torch.zeros(1), torch.zeros(1)]).bool()
    subset_holdout_mask = torch.cat([torch.zeros(pos_mask.sum().item()), torch.ones(holdout_mask.sum().item()), torch.zeros(neg_mask.sum().item()), torch.zeros(1), torch.zeros(1)]).bool()
    subset_neg_mask = torch.cat([torch.zeros(pos_mask.sum().item() + holdout_mask.sum().item()), torch.ones(neg_mask.sum().item()), torch.zeros(1), torch.zeros(1)]).bool()
    
    subset_dimred_embeddings = dim_reduce(subset_embeddings, function=reduce_fn)

    print("PLOT NUM POS:", pos_mask.sum().item())
    print("PLOT NUM HOLDOUT:", holdout_mask.sum().item())
    print("PLOT NUM NEG:", neg_mask.sum().item())

    plt.figure(figsize=(10, 10))
    plt.title(title)

    if neg_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[subset_neg_mask][~cand_mask[neg_mask]][:, 0], subset_dimred_embeddings[subset_neg_mask][~cand_mask[neg_mask]][:, 1], s=0.5, c='silver', label=f'Non-Cand. True Neg.')
        plt.scatter(subset_dimred_embeddings[subset_neg_mask][cand_mask[neg_mask]][:, 0], subset_dimred_embeddings[subset_neg_mask][cand_mask[neg_mask]][:, 1], s=0.5, c='gray', label=f'Cand. True Neg.')
    if pos_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[subset_pos_mask][:, 0], subset_dimred_embeddings[subset_pos_mask][:, 1], s=0.5, c='lightblue', label=f'Training Pos.')

    cmap = ['magenta', 'yellow', 'red', 'lime', 'orange', 'cyan', 'gray']
    for i, label in enumerate(torch.unique(labels_original[holdout_mask], dim=0)):
        label_name = [id2rel_original[relid.item()] for relid in label.nonzero().flatten()]
        plot_cand_mask = torch.logical_and((labels_original[holdout_mask] == label).all(dim=1), cand_mask[holdout_mask])
        plot_noncand_mask = torch.logical_and((labels_original[holdout_mask] == label).all(dim=1), ~cand_mask[holdout_mask])

        print(f'PLOT LABEL: {label_name} ({plot_cand_mask.sum().item()}) candidates, {plot_noncand_mask.sum().item()} non-candidates')
        plt.scatter(subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][:, 0], subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][:, 1], s=0.5, c=cmap[i], label=f'Cand. Holdout {label_name}')

    plt.scatter(subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 0], subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 1], s=0.5, c='black', label='Non-Cand. Holdout')

    if neg_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[-1, 0], subset_dimred_embeddings[-1, 1], s=70, c='red', marker='x', label=f'Neg./Holdout Mean')
    if pos_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[-2, 0], subset_dimred_embeddings[-2, 1], s=70, c='blue', marker='x', label=f'Training Pos. Mean')

    legend = plt.legend()

    for lh in legend.legendHandles:
        lh._sizes = [25]

    plt.savefig(plot_save_path)
    plt.show()