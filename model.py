import torch
import torch.nn as nn
from opt_einsum import contract
from transformers import LukeModel
import torch.nn.functional as F
import json
import const
from losses import ATLoss, SupConLoss
from encoding import encode


class DocRedModel(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_class,
                 mode,
                 contr_temp=None, # hyperparameter
                 contr_sup_weight=None, # hyperparameter
                 dual_binlog_weight=None, # hyperparameter
                 embed_size=768, # Intermediary embedding for head and tail entities
                 out_embed_size=768, # Final embedding for the relationship
                 projection_size=128,
                 max_labels=4, # Max number of classes to predict for one entity pair
                 block_size=64):
        super(DocRedModel, self).__init__()

        if model_name not in set([const.LUKE_BASE, const.LUKE_LARGE, const.LUKE_LARGE_TACRED]):
            raise ValueError(f'Invalid encoder name: {model_name}')
        
        if mode not in set([const.MODE_OFFICIAL, const.MODE_CONTRASTIVE, const.MODE_DUAL_SUPERVISED]):
            raise ValueError(f'Invalid mode: {mode}')
        
        if mode == const.MODE_CONTRASTIVE and not all([contr_temp, contr_sup_weight]):
            raise ValueError('Contrastive mode requires contrastive temperature and contrastive supervised weight')
        
        if mode == const.MODE_DUAL_SUPERVISED and not dual_binlog_weight:
            raise ValueError('Dual supervised mode requires dual binary logit weight')
        
        self.model_name = model_name

        self.start_tok_ids = [tokenizer.cls_token_id] # list of token ids that indicate the start of an input sequence
        self.end_tok_ids = [tokenizer.sep_token_id] # list of token ids that indicate the end of an input sequence

        self.embed_size = embed_size
        self.block_size = block_size
        self.hidden_size = 768 if self.model_name == const.LUKE_BASE else 1024
        self.out_embed_size = out_embed_size
        self.projection_size = projection_size

        self.mode = mode

        self.contr_temp = contr_temp
        self.contr_sup_weight = contr_sup_weight
        self.dual_binlog_weight = dual_binlog_weight

        self.atloss_fn = ATLoss()
        self.binlogloss_fn = nn.BCEWithLogitsLoss()
        self.contrloss_fn = SupConLoss(temperature=self.contr_temp)

        self.max_labels = max_labels
        self.num_class = num_class

        self.luke_model = LukeModel.from_pretrained(self.model_name) # Base model
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.embed_size)
        self.bilinear = nn.Linear(self.embed_size * self.block_size, self.out_embed_size if mode != const.MODE_OFFICIAL else self.num_class) # This is to maintain original ATLOP implementation

        self.classifier_head = nn.Linear(self.out_embed_size, self.num_class)

        self.binary_head = nn.Linear(self.out_embed_size, 1)

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
        

    def embed(self, batch):
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

        embeds = self.bilinear(bl) # returns out_embed_size or num_class based on mode

        class_logits, binary_logits, proj_logits = None, None, None

        if self.mode == const.MODE_OFFICIAL:
            class_logits = embeds # Classification head for official supervised learning -> bilinear has class logits output in official mode
            embeds = None

        elif self.mode == const.MODE_CONTRASTIVE:
            # https://github.com/hppRC/simple-simcse/blob/main/train.py#L149 -> SimCSE has tanh activation function for projection head
            # Supervised contrastive normalizes input to projection head but SimCLR does not
            proj_logits = torch.tanh(self.projection_head(F.normalize(embeds, dim=-1))) # normalizing before projection input seems to be good
        
        elif self.mode == const.MODE_DUAL_SUPERVISED:
            class_logits = self.classifier_head(embeds)
            binary_logits = self.binary_head(embeds)
        
        return embeds, class_logits, binary_logits, proj_logits

        


    def forward(self,
                batch,
                train):
        labels = [torch.tensor(l) for l in batch['labels']]
        labels = torch.cat(labels, dim=0).float()

        embeds, preds = None, None
        update_loss, sup_loss, contr_loss, sup_contr_loss, unsup_contr_loss = None, None, None, None, None 

        if train: # Training
            if self.mode == const.MODE_CONTRASTIVE:
                embeds, _, _, proj_logits1 = self.embed(batch)
                _, _, _, proj_logits2 = self.embed(batch) # Forward pass 2 - Done for ensuring different dropout masks are applied (sufficient augmentation)

                proj_logits = torch.stack([proj_logits1, proj_logits2], dim=1) # [batch_size, 2, projection_size]
                proj_logits = F.normalize(proj_logits, dim=-1) # Have to normalize per https://github.com/HobbitLong/SupContrast/issues/22 for SupConLoss. Look at various questions/issues on 'normalization and nan loss'

                contr_labels = (labels[:, 0] == 1).long()

                sup_contr_loss = self.contrloss_fn(features=proj_logits, labels=contr_labels)
                unsup_contr_loss = self.contrloss_fn(features=proj_logits)

                contr_loss = (self.contr_sup_weight * sup_contr_loss) + ((1 - self.contr_sup_weight) * unsup_contr_loss) # previous plotting was done with 0.45/0.55

                update_loss = contr_loss

            elif self.mode == const.MODE_OFFICIAL: 
                _, class_logits, _, _ = self.embed(batch)

                sup_loss = self.atloss_fn(class_logits.float(), labels.float().to(const.DEVICE))
                
                update_loss = sup_loss

            elif self.mode == const.MODE_DUAL_SUPERVISED:
                embeds, class_logits, binary_logits, _ = self.embed(batch)

                binlog_labels = (labels[:, 0] == 1).float()

                sup_loss = self.atloss_fn(class_logits.float(), labels.float().to(const.DEVICE))
                binlog_loss = self.binlogloss_fn(binary_logits.flatten().float(), binlog_labels.float().to(const.DEVICE)) # NOTE: double check this

                dual_loss = (self.dual_binlog_weight * binlog_loss) + ((1 - self.dual_binlog_weight) * sup_loss)

                update_loss = dual_loss

            if update_loss is None:
                raise ValueError('Loss to update is None')
            
        else: # Validation
            if self.mode == const.MODE_OFFICIAL:
                _, class_logits, _, _ = self.embed(batch)
                preds = self.atloss_fn.get_label(class_logits.float(), num_labels=self.max_labels)
                embeds = torch.zeros(class_logits.size(0), self.out_embed_size).to(const.DEVICE) # No embeddings for official mode

            elif self.mode == const.MODE_CONTRASTIVE:
                embeds, _, _, _ = self.embed(batch)
                preds = torch.zeros(embeds.size(0), self.num_class).to(const.DEVICE) # No predictions for contrastive mode

            elif self.mode == const.MODE_DUAL_SUPERVISED:
                embeds, class_logits, binary_logits, _ = self.embed(batch)
                preds = self.atloss_fn.get_label(class_logits.float(), num_labels=self.max_labels)
                # TODO: add binary logits to predictions

            if preds is None or embeds is None:
                raise ValueError('Prediction or embeddings are None')

        losses = {
            'update_loss': update_loss if update_loss else torch.tensor(-1).to(const.DEVICE), # Dynamically update loss based on mode
            'sup_loss': sup_loss if sup_loss else torch.tensor(-1).to(const.DEVICE),
            'contr_loss': contr_loss if contr_loss else torch.tensor(-1).to(const.DEVICE),
            'sup_contr_loss': sup_contr_loss if sup_contr_loss else torch.tensor(-1).to(const.DEVICE),
            'unsup_contr_loss': unsup_contr_loss if unsup_contr_loss else torch.tensor(-1).to(const.DEVICE)
        }

        return embeds, preds, losses