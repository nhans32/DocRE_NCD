from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from scipy.spatial import ConvexHull

def dim_reduce(embeddings,
               n_components=2,
               n_neighbors=30,
               function='umap',
               densmap=False):
    print(f"Dimension Reducing ({function})...")
    if function == 'umap':
        dimred = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0, densmap=densmap, n_jobs=-1)
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
                     num=100_000,
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

    elif mode == 'neg_pos_holdouts':
        subset_pos_mask = pos_mask
        subset_holdout_mask = holdout_mask
        subset_neg_mask = torch.zeros(neg_mask.size(0)).bool()

        num_negs = num - pos_mask.sum().item() - holdout_mask.sum().item()
        if num_negs > 0:
            neg_mask_nonzeros = neg_mask.nonzero().flatten()
            subset_neg_mask[neg_mask_nonzeros[torch.randperm(neg_mask_nonzeros.size(0))[:num_negs]]] = True

    elif mode == 'neg_holdouts':
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
         cluster_labels=None,
         densmap=False): # Densmap for plotting?

    pos_mean = embeddings[pos_mask].mean(dim=0).unsqueeze(0) if pos_mask.sum().item() != 0 else embeddings[neg_mask][0].unsqueeze(0) # have placeholder
    neg_mean = embeddings[torch.logical_or(holdout_mask, neg_mask)].mean(dim=0).unsqueeze(0) if torch.logical_or(holdout_mask, neg_mask).sum().item() != 0 else embeddings[pos_mask][0].unsqueeze(0) 

    subset_mask = torch.logical_or(pos_mask, torch.logical_or(holdout_mask, neg_mask))
    subset_embeddings = torch.cat([embeddings[subset_mask], pos_mean, neg_mean], dim=0)

    subset_pos_mask = torch.cat([pos_mask[subset_mask], torch.zeros(1).bool(), torch.zeros(1).bool()], dim=0)
    subset_holdout_mask = torch.cat([holdout_mask[subset_mask], torch.zeros(1).bool(), torch.zeros(1).bool()], dim=0)
    subset_neg_mask = torch.cat([neg_mask[subset_mask], torch.zeros(1).bool(), torch.zeros(1).bool()], dim=0)
    
    subset_dimred_embeddings = dim_reduce(subset_embeddings, function='umap', densmap=densmap)

    print('-=-=-=PLOT INFO=-=-=-')
    print("NUM. POS:", pos_mask.sum().item())
    print("NUM. HOLDOUT:", holdout_mask.sum().item())
    print("NUM. NEG:", neg_mask.sum().item())
    print('-=-=-=-')

    plt.figure(figsize=(10, 10))
    plt.title(title)

    if neg_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[subset_neg_mask][~cand_mask[neg_mask]][:, 0], subset_dimred_embeddings[subset_neg_mask][~cand_mask[neg_mask]][:, 1], s=0.5, c='silver', label=f'Non-Cand. True Neg.')
        plt.scatter(subset_dimred_embeddings[subset_neg_mask][cand_mask[neg_mask]][:, 0], subset_dimred_embeddings[subset_neg_mask][cand_mask[neg_mask]][:, 1], s=0.5, c='gray', label=f'Cand. True Neg.')
    if pos_mask.sum().item() > 0:
        plt.scatter(subset_dimred_embeddings[subset_pos_mask][:, 0], subset_dimred_embeddings[subset_pos_mask][:, 1], s=0.5, c='lightblue', label=f'Training Pos.')

    if cluster_labels is not None:
        plt.scatter(subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 0], subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 1], s=0.5, c='black', label='Non-Cand. Holdout')

    cmap = ['magenta', 'yellow', 'red', 'lime', 'orange', 'cyan', 'gold', 'pink', 'purple', 'brown', 'blue', 'green']
    seen_rels = set()
    for i, label in enumerate(torch.unique(labels_original[holdout_mask], dim=0)):
        for relid in label.nonzero().flatten():
            if relid.item() in seen_rels:
                continue

            label_name = id2rel_original[relid.item()]
            plot_cand_mask = torch.logical_and(labels_original[holdout_mask][:, relid.item()] == 1, cand_mask[holdout_mask])
            plot_noncand_mask = torch.logical_and(labels_original[holdout_mask][:, relid.item()] == 1, ~cand_mask[holdout_mask])

            print(f'LABEL: {label_name} {plot_cand_mask.sum().item()}/{plot_cand_mask.sum().item() + plot_noncand_mask.sum().item()} ({plot_cand_mask.sum().item()/(plot_cand_mask.sum().item() + plot_noncand_mask.sum().item()):.2f}) {"candidates" if cluster_labels is None else "clustered"}')

            if plot_cand_mask.sum().item() != 0:
                if cluster_labels is None:
                    plt.scatter(subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][:, 0], subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][:, 1], s=0.5, c=cmap[i], label=f'Cand. Holdout {label_name}')
                else:
                    max_clust = cluster_labels[holdout_mask][plot_cand_mask].mode().values.item()
                    clust_mask = cluster_labels[holdout_mask][plot_cand_mask] == max_clust
                    print(f'{clust_mask.sum().item()}/{plot_cand_mask.sum().item()} in cluster {max_clust} (size: {(cluster_labels == max_clust).sum().item()}, purity: {(clust_mask.sum().item() / (cluster_labels == max_clust).sum().item()):.2f})')
                    plt.scatter(subset_dimred_embeddings[subset_neg_mask][cluster_labels[neg_mask] == max_clust][:, 0], subset_dimred_embeddings[subset_neg_mask][cluster_labels[neg_mask] == max_clust][:, 1], s=0.5, c='lightskyblue')
                    plt.scatter(subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][~clust_mask][:, 0], subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][~clust_mask][:, 1], s=0.5, c='brown')
                    plt.scatter(subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][clust_mask][:, 0], subset_dimred_embeddings[subset_holdout_mask][plot_cand_mask][clust_mask][:, 1], s=0.5, c=cmap[i], label=f'Cand. Holdout {label_name} - Clust. {max_clust}')
            seen_rels.add(relid.item())
            print('-----')

    if cluster_labels is None:
        plt.scatter(subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 0], subset_dimred_embeddings[subset_holdout_mask][~cand_mask[holdout_mask]][:, 1], s=0.5, c='black', label='Non-Cand. Holdout')
        
    if neg_mask.sum().item() > 0 and cluster_labels is None:
        plt.scatter(subset_dimred_embeddings[-1, 0], subset_dimred_embeddings[-1, 1], s=70, c='red', marker='x', label=f'Neg./Holdout Mean')
    if pos_mask.sum().item() > 0 and cluster_labels is None:
        plt.scatter(subset_dimred_embeddings[-2, 0], subset_dimred_embeddings[-2, 1], s=70, c='blue', marker='x', label=f'Training Pos. Mean')

    legend = plt.legend()

    for lh in legend.legendHandles:
        lh._sizes = [25]

    plt.savefig(plot_save_path)
    plt.show()