# ADAPTED FROM: https://github.com/wzhouad/ATLOP/blob/main/long_seq.py

import torch
import numpy as np
import torch.nn.functional as F
import const

def encode(model,
           batch,
           start_tok_ids,
           end_tok_ids):
    
    start_tok_ids = torch.tensor(start_tok_ids).to(batch['input_ids'])
    end_tok_ids = torch.tensor(end_tok_ids).to(batch['input_ids'])

    seq_len = batch['input_ids'].size(-1)
    entity_len = batch['entity_ids'].size(-1)

    if seq_len <= const.MAX_ENCODER_LENGTH:
        # If batch sequence length in the batch is less than the max encoder length, we can just encode the batch as is.
        output = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            entity_ids=batch['entity_ids'],
            entity_attention_mask=batch['entity_attention_mask'],
            entity_position_ids=batch['entity_position_ids'],
            output_attentions=True
        )

        seq_lhs = output['last_hidden_state']
        ent_lhs = output['entity_last_hidden_state']

        attn = output['attentions'][-1] # Only getting last layer of attention [batch_n, num_heads, seq_len + ent_len, seq_len + ent_len]. NOTE: Could expand this to last n layers (per other papers).
        ent_to_seq_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), :seq_lhs.shape[1]] # Get entity to base sequence attention values
        ent_to_ent_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1])] # Get entity to entity attention values
        
        entity_id_labels = batch['entity_id_labels']

        return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels
    else:
        # If batch sequence length in the batch is greater than the max encoder length, we need to split the batch into smaller chunks.
        seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels = long_encode(model=model,
                                                                                           batch=batch,
                                                                                           start_tok_ids=start_tok_ids,
                                                                                           end_tok_ids=end_tok_ids,
                                                                                           seq_len=seq_len,
                                                                                           entity_len=entity_len)
        return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels
        

def long_encode(model,
                batch,
                start_tok_ids,
                end_tok_ids,
                seq_len,
                entity_len):
    
    new_input_ids, new_attention_mask, new_entity_ids, new_entity_attention_mask, new_entity_position_ids, new_entity_id_labels = [], [], [], [], [], []

    nonpad_doc_lens = batch['attention_mask'].sum(1).cpu().numpy().astype(np.int32).tolist() # Length of each document in the batch without padding
    num_segments = [] # How many chunks for each document

    for doc_idx, doc_len in enumerate(nonpad_doc_lens): # Go through every document in the batch, checking if it needs to be chunked
        if doc_len <= const.MAX_ENCODER_LENGTH: # No need to chunk
            new_input_ids.append(batch['input_ids'][doc_idx])
            new_attention_mask.append(batch['attention_mask'][doc_idx])
            new_entity_ids.append(batch['entity_ids'][doc_idx])
            new_entity_attention_mask.append(batch['entity_attention_mask'][doc_idx])
            new_entity_position_ids.append(batch['entity_position_ids'][doc_idx])
            new_entity_id_labels.append(batch['entity_id_labels'][doc_idx])

            num_segments.append(1)
        else: # Must chunk into two segments
            input_ids1 = torch.cat([batch['input_ids'][doc_idx, :const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)], end_tok_ids], dim=-1) # First half token sequence, from start to max encoder length
            attention_mask1 = batch['attention_mask'][doc_idx, :const.MAX_ENCODER_LENGTH]

            input_ids2 = torch.cat([start_tok_ids, batch['input_ids'][doc_idx, (doc_len - const.MAX_ENCODER_LENGTH + start_tok_ids.size(0)) : doc_len]], dim=-1) # Second half token sequence, from end to max encoder length
            attention_mask2 =  batch['attention_mask'][doc_idx, (doc_len - const.MAX_ENCODER_LENGTH) : doc_len]

            entity_ids1, entity_attention_mask1, entity_position_ids1, entity_id_labels1 = [], [], [], [] # First segment
            entity_ids2, entity_attention_mask2, entity_position_ids2, entity_id_labels2 = [], [], [], [] # Second segment

            for ment_idx, entity_id in enumerate(batch['entity_id_labels'][doc_idx]):
                ment_tok_start = batch['entity_position_ids'][doc_idx][ment_idx][0] # Start position of entity mention in the document

                if ment_tok_start < (const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)): # Entity in the first segment
                    entity_ids1.append(batch['entity_ids'][doc_idx][ment_idx])
                    entity_attention_mask1.append(batch['entity_attention_mask'][doc_idx][ment_idx])
                    entity_position_ids1.append(batch['entity_position_ids'][doc_idx][ment_idx])
                    entity_position_ids1[-1][entity_position_ids1[-1] >= (const.MAX_ENCODER_LENGTH - end_tok_ids.size(0))] = const.PAD_IDS['entity_position_ids'] # Set out of bounds positions to -1
                    entity_id_labels1.append(entity_id)

                if ment_tok_start > (doc_len - const.MAX_ENCODER_LENGTH) and ment_tok_start < (const.MAX_DOC_LENGTH - end_tok_ids.size(0)): # Entity in the second segment


            
            num_segments.append(2)
