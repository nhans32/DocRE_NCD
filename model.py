import torch
import torch.nn as nn
from opt_einsum import contract
from transformers import LukeModel
import torch.nn.functional as F
import json
import const
from losses import ATLoss
from encoding import encode

class SupConLoss(torch.nn.Module): # From: https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/contrastive_training/contrastive_training.py
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device(const.DEVICE)
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class DocRedModel(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_class,
                 contrastive_tmp,
                 embed_size=768, # Intermediary embedding for head and tail entities
                 out_embed_size=768, # Final embedding for the relationship
                 projection_size=128,
                 max_labels=4, # Max number of classes to predict for one entity pair
                 block_size=64):
        super(DocRedModel, self).__init__()

        if model_name not in set([const.LUKE_BASE, const.LUKE_LARGE, const.LUKE_LARGE_TACRED]):
            raise ValueError(f'Invalid encoder name: {model_name}')
        self.model_name = model_name

        self.start_tok_ids = [tokenizer.cls_token_id] # list of token ids that indicate the start of an input sequence
        self.end_tok_ids = [tokenizer.sep_token_id] # list of token ids that indicate the end of an input sequence

        self.embed_size = embed_size
        self.block_size = block_size
        self.hidden_size = 768 if self.model_name == const.LUKE_BASE else 1024
        self.out_embed_size = out_embed_size
        self.projection_size = projection_size

        self.atloss_fn = ATLoss()
        self.posclassloss_fn = nn.BCEWithLogitsLoss()
        self.contrloss_fn = SupConLoss(temperature=contrastive_tmp)

        self.contrastive_tmp = contrastive_tmp

        self.max_labels = max_labels
        self.num_class = num_class

        self.luke_model = LukeModel.from_pretrained(self.model_name) # Base model
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.bilinear = nn.Linear(self.embed_size * self.block_size, self.out_embed_size)

        self.classifier_head = nn.Linear(self.out_embed_size, self.num_class)

        self.projection_head = nn.Linear(self.out_embed_size, self.projection_size)


    def hrt_pool(self,
                 seq_lhs,
                 ent_lhs,
                 ent_to_seq_attn,
                 ent_to_ent_attn,
                 entity_id_labels,
                 entity_pos,
                 hts):
        batch_size, num_attn_heads, _, seq_len = ent_to_seq_attn.size()
        hss, tss, rel_seq_embeds = [], [], []

        for doc_i in range(batch_size): # For every document in batch
            cur_entity_id_labels = torch.tensor(entity_id_labels[doc_i]) # Get the tensor slices for the document entities, removing padding
            cur_ent_lhs = ent_lhs[doc_i][:len(cur_entity_id_labels)] # Only need lhs from entity mentions not padding (cur_entity_id_labels is not padded so it gives us the amount of entity mentions)
            cur_ent_to_seq_attn = ent_to_seq_attn[doc_i][:, :len(cur_entity_id_labels), :] # [heads, n_ment, seq_len]

            ent_embeds, ent_seq_attn = [], []
            for ent_i in range(len(entity_pos[doc_i])): # For every entity in the document NOTE: (entity NOT mention). entity_pos is a list of lists of mentions for each entity [[mention1, mention2], [mention1], ...]
                e_mask_i = cur_entity_id_labels == ent_i # Apply entity mask to get embeddings for entity e_i's mentions
                e_emb = cur_ent_lhs[e_mask_i]
                e_ent_seq_attn = cur_ent_to_seq_attn[:, e_mask_i, :]

                if len(e_emb) > 1:
                    e_emb = torch.logsumexp(e_emb, dim=0) # Logsumexp pool mention representations: [n_ment, hidden_size] -> [hidden_size]
                    e_ent_seq_attn = e_ent_seq_attn.mean(1) # Average base sequence attentions across e_i's mentions: [heads, n_ment, seq_len] -> [heads, seq_len]
                elif len(e_emb) == 1: # No need to pool if only one mention
                    e_emb = e_emb[0]
                    e_ent_seq_attn = e_ent_seq_attn[:, 0, :]
                else: # If no mentions for entity ent_i, zero out
                    e_emb = torch.zeros(self.hidden_size).to(ent_lhs)
                    e_ent_seq_attn = torch.zeros(num_attn_heads, seq_len).to(ent_to_seq_attn)

                ent_embeds.append(e_emb)
                ent_seq_attn.append(e_ent_seq_attn)
            
            ent_embeds = torch.stack(ent_embeds, dim=0) # [n_ent, hidden_size]
            ent_seq_attn = torch.stack(ent_seq_attn, dim=0) # [n_ent, heads, seq_len]

            # Head-Tail Pairs
            ht_i = torch.LongTensor(hts[doc_i]).to(seq_lhs.device)

            # Get HT Entity Embeddings
            hs = torch.index_select(ent_embeds, 0, ht_i[:, 0])
            ts = torch.index_select(ent_embeds, 0, ht_i[:, 1])

            # Base Sequence Attention Embedding Calculation
            h_seq_attn = torch.index_select(ent_seq_attn, 0, ht_i[:, 0])
            t_seq_attn = torch.index_select(ent_seq_attn, 0, ht_i[:, 1])
            ht_seq_attn = (h_seq_attn * t_seq_attn).mean(1) # Multiply the head and tail attentions and take the mean
            ht_seq_attn = ht_seq_attn / (ht_seq_attn.sum(1, keepdim=True) + 1e-5)
            rel_seq_emb = contract("ld,rl->rd", seq_lhs[doc_i], ht_seq_attn) # NOTE: explain what contract does

            hss.append(hs)
            tss.append(ts)
            rel_seq_embeds.append(rel_seq_emb)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rel_seq_embeds = torch.cat(rel_seq_embeds, dim=0) # Representation of relationship between h,t based on base sequence

        return hss, tss, rel_seq_embeds
        

    def embed(self, batch, mode):
        seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels = encode(model=self.luke_model,
                                                                                      batch=batch,
                                                                                      start_tok_ids=self.start_tok_ids,
                                                                                      end_tok_ids=self.end_tok_ids)
        # NOTE: Keep in mind that each "entity" at this point (after collate_fn) is simply a single mention of an entity.
        #       A single mention of an entity does not necessarily represent the entire entity (there can be multiple mentions).
        #       Therefore, we pool entity mentions in the hrt_pool function to operate with single entity representations.
        hs, ts, rel_seq_embeds = self.hrt_pool(seq_lhs=seq_lhs,
                                               ent_lhs=ent_lhs,
                                               ent_to_seq_attn=ent_to_seq_attn,
                                               ent_to_ent_attn=ent_to_ent_attn,
                                               entity_id_labels=entity_id_labels,
                                               entity_pos=batch['entity_pos'],
                                               hts=batch['hts'])

        hs = torch.cat([hs, rel_seq_embeds], dim=1)
        ts = torch.cat([ts, rel_seq_embeds], dim=1)

        hs = torch.tanh(self.head_extractor(hs))
        ts = torch.tanh(self.tail_extractor(ts))

        b1 = hs.view(-1, self.embed_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.embed_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.embed_size * self.block_size)

        embeds = self.bilinear(bl)

        class_logits, proj_logits = None, None

        if mode == const.MODE_SUPERVISED:
            class_logits = self.classifier_head(embeds) # Classification head for supervised learning

        if mode == const.MODE_CONTRASTIVE:
            proj_logits = torch.tanh(self.projection_head(embeds)) # Projection head for SimCSE

        return embeds, class_logits, proj_logits
    

    def forward(self,
                batch,
                mode,
                train):
        
        if mode not in set([const.MODE_SUPERVISED, const.MODE_CONTRASTIVE]):
            raise ValueError(f'Invalid mode: {mode}')

        labels = [torch.tensor(l) for l in batch['labels']]
        labels = torch.cat(labels, dim=0).float()

        embeds, preds = None, None
        update_loss, sup_loss, contr_loss = None, None, None    

        if train: # Training
            if mode == const.MODE_CONTRASTIVE:
                embeds, _, proj_logits1 = self.embed(batch, mode)
                _, _, proj_logits2 = self.embed(batch, mode) # Forward pass 2 - Done for ensuring different dropout masks are applied (sufficient augmentation)

                proj_logits = torch.stack([proj_logits1, proj_logits2], dim=1) # [batch_size, 2, projection_size]
                contr_labels = (labels[:, 0] == 1).long() # Positive pairs have the same label, negative pairs have same label
                contr_loss = self.contrloss_fn(features=proj_logits, labels=contr_labels)
                
                # Old simple SimCLR
                # sim_matrix = F.cosine_similarity(proj_logits1.unsqueeze(1), proj_logits2.unsqueeze(0), dim=-1) # Calculate cosine similarity between all pairs of embeddings
                # sim_matrix = sim_matrix / self.contrastive_tmp
                # contr_labels = torch.arange(sim_matrix.size(0)).long().to(const.DEVICE)

                # contr_loss = self.contrloss_fn(sim_matrix, contr_labels)

                update_loss = contr_loss

            elif mode == const.MODE_SUPERVISED: 
                embeds, class_logits, _ = self.embed(batch, mode)

                sup_loss = self.atloss_fn(class_logits.float(), labels.float().to(const.DEVICE))
                
                update_loss = sup_loss

            if update_loss is None:
                raise ValueError('Loss to update is None')
            
        else: # Validation
            if mode == const.MODE_SUPERVISED:
                embeds, class_logits, _ = self.embed(batch, mode)
                preds = self.atloss_fn.get_label(class_logits.float(), num_labels=self.max_labels)

            elif mode == const.MODE_CONTRASTIVE:
                embeds, _, _ = self.embed(batch, mode)
                preds = torch.zeros(embeds.size(0), self.num_class).to(const.DEVICE) # No predictions for contrastive mode

            if preds is None:
                raise ValueError('Predictions are None')

        losses = {
            'update_loss': update_loss if update_loss else torch.tensor(-1).to(const.DEVICE), # Dynamically update loss based on mode
            'sup_loss': sup_loss if sup_loss else torch.tensor(-1).to(const.DEVICE),
            'contr_loss': contr_loss if contr_loss else torch.tensor(-1).to(const.DEVICE)
        }

        return embeds, preds, losses