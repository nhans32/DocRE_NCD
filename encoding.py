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

    batch_seq_len = batch['input_ids'].size(-1)
    batch_entity_len = batch['entity_ids'].size(-1)

    if batch_seq_len <= const.MAX_ENCODER_LENGTH:
        # If batch sequence length in the batch is less than the max encoder length, we can just encode the batch as is.
        output = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            entity_ids=batch['entity_ids'],
            entity_attention_mask=batch['entity_attention_mask'],
            entity_position_ids=batch['entity_position_ids'],
            output_attentions=True)

        seq_lhs = output['last_hidden_state']
        ent_lhs = output['entity_last_hidden_state']

        attn = output['attentions'][-1] # Only getting last layer of attention [batch_n, num_heads, seq_len + ent_len, seq_len + ent_len]. NOTE: Could expand this to last n layers (per other papers).
        ent_to_seq_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), :seq_lhs.shape[1]] # Get entity to base sequence attention values
        ent_to_ent_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1])] # Get entity to entity attention values
        
        entity_id_labels = batch['entity_id_labels']

        return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels
    else:
        # If batch sequence length in the batch is greater than the max encoder length, we need to split the documents in the batch into overlapping chunks.
        seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels = long_encode(model=model,
                                                                                           batch=batch,
                                                                                           start_tok_ids=start_tok_ids,
                                                                                           end_tok_ids=end_tok_ids,
                                                                                           batch_seq_len=batch_seq_len,
                                                                                           batch_entity_len=batch_entity_len)
        return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels
        

def long_encode(model,
                batch,
                start_tok_ids,
                end_tok_ids,
                batch_seq_len,
                batch_entity_len):
    
    new_input_ids, new_attention_mask, new_entity_ids, new_entity_attention_mask, new_entity_position_ids, new_entity_id_labels = [], [], [], [], [], []

    doc_lens = batch['attention_mask'].sum(1).cpu().numpy().astype(np.int32).tolist() # Length of each document in the batch without padding
    doc_segments = [] # How many chunks for each document

    for doc_i, doc_len in enumerate(doc_lens): # Go through every document in the batch, checking if it needs to be chunked
        if doc_len <= const.MAX_ENCODER_LENGTH: # No need to chunk
            new_input_ids.append(batch['input_ids'][doc_i])
            new_attention_mask.append(batch['attention_mask'][doc_i])
            new_entity_ids.append(batch['entity_ids'][doc_i])
            new_entity_attention_mask.append(batch['entity_attention_mask'][doc_i])
            new_entity_position_ids.append(batch['entity_position_ids'][doc_i])
            new_entity_id_labels.append(batch['entity_id_labels'][doc_i])

            doc_segments.append(1)
        else: # Must chunk into two segments
            input_ids1 = torch.cat([batch['input_ids'][doc_i, :const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)], end_tok_ids], dim=-1) # First half token sequence, from start to max encoder length
            attention_mask1 = batch['attention_mask'][doc_i, :const.MAX_ENCODER_LENGTH]

            input_ids2 = torch.cat([start_tok_ids, batch['input_ids'][doc_i, (doc_len - const.MAX_ENCODER_LENGTH + start_tok_ids.size(0)) : doc_len]], dim=-1) # Second half token sequence, from end to max encoder length
            attention_mask2 =  batch['attention_mask'][doc_i, (doc_len - const.MAX_ENCODER_LENGTH) : doc_len]

            entity_ids1, entity_attention_mask1, entity_position_ids1, entity_id_labels1 = [], [], [], [] # First segment
            entity_ids2, entity_attention_mask2, entity_position_ids2, entity_id_labels2 = [], [], [], [] # Second segment
            for ment_i, entity_id in enumerate(batch['entity_id_labels'][doc_i]):
                ment_tok_start = batch['entity_position_ids'][doc_i][ment_i][0] # Start position of entity mention in the document

                if ment_tok_start < (const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)): # Entity in the first segment
                    entity_ids1.append(batch['entity_ids'][doc_i][ment_i])
                    entity_attention_mask1.append(batch['entity_attention_mask'][doc_i][ment_i])
                    entity_position_ids1.append(batch['entity_position_ids'][doc_i][ment_i])
                    entity_position_ids1[-1][entity_position_ids1[-1] >= (const.MAX_ENCODER_LENGTH - end_tok_ids.size(0))] = const.PAD_IDS['entity_position_ids'] # Set out of bounds positions to -1
                    entity_id_labels1.append(entity_id)

                if ment_tok_start > (doc_len - const.MAX_ENCODER_LENGTH) and ment_tok_start < (const.MAX_DOC_LENGTH - end_tok_ids.size(0)): # Entity in the second segment
                    entity_ids2.append(batch['entity_ids'][doc_i][ment_i])
                    entity_attention_mask2.append(batch['entity_attention_mask'][doc_i][ment_i])
                    entity_position_ids2.append(batch['entity_position_ids'][doc_i][ment_i])
                    entity_position_ids2[-1] -= (doc_len - const.MAX_ENCODER_LENGTH) # Reorienting position indexes with second segment start indexes. This will set -1s to below -1, adjusting for this in the subsequent line
                    entity_position_ids2[-1][torch.logical_or(entity_position_ids2[-1] >= (const.MAX_DOC_LENGTH - end_tok_ids.size(0)), entity_position_ids2[-1] < 0)] = const.PAD_IDS['entity_position_ids'] # set out of bounds positions to -1
                    entity_id_labels2.append(entity_id)

                # else: Entity is in neither segment, ignore it. 
                # NOTE: Is there a way to handle things when spans are split across segments? (e.g. entity mention spans from first segment to second segment)

            entity_ids1 = torch.tensor(entity_ids1).to(batch['entity_ids'])
            entity_attention_mask1 = torch.tensor(entity_attention_mask1).to(batch['entity_attention_mask'])
            entity_position_ids1 = torch.tensor(entity_position_ids1).to(batch['entity_position_ids'])

            entity_ids2 = torch.tensor(entity_ids2).to(batch['entity_ids'])
            entity_attention_mask2 = torch.tensor(entity_attention_mask2).to(batch['entity_attention_mask'])
            entity_position_ids2 = torch.tensor(entity_position_ids2).to(batch['entity_position_ids'])

            entity_ids1 = F.pad(entity_ids1, (0, batch_entity_len - len(entity_ids1)), 'constant', const.PAD_IDS['entity_ids'])
            entity_attention_mask1 = F.pad(entity_attention_mask1, (0, batch_entity_len - len(entity_attention_mask1)), 'constant', const.PAD_IDS['entity_attention_mask'])
            entity_position_ids1 = F.pad(entity_position_ids1, (0, 0, 0, batch_entity_len - len(entity_position_ids1)), 'constant', const.PAD_IDS['entity_position_ids'])

            entity_ids2 = F.pad(entity_ids2, (0, batch_entity_len - len(entity_ids2)), 'constant', const.PAD_IDS['entity_ids'])
            entity_attention_mask2 = F.pad(entity_attention_mask2, (0, batch_entity_len - len(entity_attention_mask2)), 'constant', const.PAD_IDS['entity_attention_mask'])
            entity_position_ids2 = F.pad(entity_position_ids2, (0, 0, 0, batch_entity_len - len(entity_position_ids2)), 'constant', const.PAD_IDS['entity_position_ids'])

            new_input_ids.extend([input_ids1, input_ids2])
            new_attention_mask.extend([attention_mask1, attention_mask2])
            new_entity_ids.extend([entity_ids1, entity_ids2])
            new_entity_attention_mask.extend([entity_attention_mask1, entity_attention_mask2])
            new_entity_position_ids.extend([entity_position_ids1, entity_position_ids2])
            new_entity_id_labels.extend([entity_id_labels1, entity_id_labels2])

            doc_segments.append(2)

    input_ids = torch.stack(new_input_ids, dim=0)
    attention_mask = torch.stack(new_attention_mask, dim=0)
    entity_ids = torch.stack(new_entity_ids, dim=0)
    entity_attention_mask = torch.stack(new_entity_attention_mask, dim=0)
    entity_position_ids = torch.stack(new_entity_position_ids, dim=0)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        entity_attention_mask=entity_attention_mask,
        entity_position_ids=entity_position_ids,
        output_attentions=True)

    seq_lhs = output['last_hidden_state']
    ent_lhs = output['entity_last_hidden_state']

    attn = output['attentions'][-1] # Only getting last layer of attention [batch_n, num_heads, seq_len + ent_len, seq_len + ent_len]. NOTE: Could expand this to last n layers (per other papers).
    ent_to_seq_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), :seq_lhs.shape[1]] # Get entity to base sequence attention values
    ent_to_ent_attn = attn[:, :, seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1]), seq_lhs.shape[1]:(seq_lhs.shape[1] + ent_lhs.shape[1])] # Get entity to entity attention values
    
    entity_id_labels = new_entity_id_labels

    seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels = recombine_chunks(doc_lens=doc_lens,
                                                                                            doc_segments=doc_segments,
                                                                                            attention_mask=attention_mask,
                                                                                            seq_lhs=seq_lhs,
                                                                                            ent_lhs=ent_lhs,
                                                                                            ent_to_seq_attn=ent_to_seq_attn,
                                                                                            ent_to_ent_attn=ent_to_ent_attn,
                                                                                            entity_id_labels=entity_id_labels,
                                                                                            start_tok_ids=start_tok_ids,
                                                                                            end_tok_ids=end_tok_ids,
                                                                                            batch_seq_len=batch_seq_len,
                                                                                            batch_entity_len=batch_entity_len)
    return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels



def recombine_chunks(doc_lens,
                     doc_segments,
                     attention_mask,
                     seq_lhs,
                     ent_lhs, 
                     ent_to_seq_attn,
                     ent_to_ent_attn,
                     entity_id_labels,
                     start_tok_ids,
                     end_tok_ids,
                     batch_seq_len,
                     batch_entity_len):
    i = 0
    new_entity_len = batch_entity_len
    new_seq_lhs, new_ent_lhs, new_ent_to_seq_attn, new_ent_to_ent_attn, new_entity_id_labels = [], [], [], [], []
    
    for (doc_seg, doc_seq_len) in zip(doc_segments, doc_lens):
        if doc_seg == 1:
            new_seq_lhs.append(F.pad(seq_lhs[i], (0, 0, 0, batch_seq_len - const.MAX_ENCODER_LENGTH)))
            new_ent_lhs.append(ent_lhs[i])

            new_ent_to_seq_attn.append(F.pad(ent_to_seq_attn[i], (0, batch_seq_len - const.MAX_ENCODER_LENGTH)))
            new_ent_to_ent_attn.append(ent_to_ent_attn[i])

            new_entity_id_labels.append(entity_id_labels[i])

            i += 1 # Skip to next document (1 because this document was in a single segment)
        elif doc_seg == 2:
            entity_id_labels_combined = entity_id_labels[i] + entity_id_labels[i + 1]
            new_entity_len = max(new_entity_len, len(entity_id_labels_combined))

            # -- ATTENTION MASK COMBINATION --
            attn_mask1 = attention_mask[i][:const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)]
            attn_mask1 = F.pad(attn_mask1, (0, batch_seq_len - const.MAX_ENCODER_LENGTH + end_tok_ids.size(0)))
            attn_mask2 = attention_mask[i + 1][start_tok_ids.size(0):]
            attn_mask2 = F.pad(attn_mask2, (doc_seq_len - const.MAX_ENCODER_LENGTH + start_tok_ids.size(0), batch_seq_len - doc_seq_len))
            attn_mask = attn_mask1 + attn_mask2 + 1e-10

            # -- BASE SEQUENCE HIDDEN STATES COMBINATION --
            seq_hs1 = seq_lhs[i][:const.MAX_ENCODER_LENGTH - end_tok_ids.size(0)]
            seq_hs1 = F.pad(seq_hs1, (0, 0, 0, batch_seq_len - const.MAX_ENCODER_LENGTH + end_tok_ids.size(0)))
            seq_hs2 = seq_lhs[i + 1][start_tok_ids.size(0):]
            seq_hs2 = F.pad(seq_hs2, (0, 0, doc_seq_len - const.MAX_ENCODER_LENGTH + start_tok_ids.size(0), batch_seq_len - doc_seq_len))
            seq_hs = (seq_hs1 + seq_hs2) / attn_mask.unsqueeze(-1)

            # -- ENTITY HIDDEN STATES COMBINATION --
            ent_hs = torch.cat([ent_lhs[i][:len(entity_id_labels[i])], ent_lhs[i + 1][:len(entity_id_labels[i + 1])]], dim=0)
            ent_hs = F.pad(ent_hs, (0, 0, 0, new_entity_len - len(entity_id_labels_combined))) # Pad to maximum entity length compared to original entity length

            # -- ENTITY TO BASE SEQUENCE ATTENTION COMBINATION --
            ent_seq_attn1 = ent_to_seq_attn[i][:, :len(entity_id_labels[i]), :const.MAX_ENCODER_LEN - end_tok_ids.size(0)]
            ent_seq_attn1 = F.pad(ent_seq_attn1, (0, batch_seq_len - const.MAX_ENCODER_LENGTH + end_tok_ids.size(0), 0, new_entity_len - len(entity_id_labels[i])))

            ent_seq_attn2 = ent_to_seq_attn[i + 1][:, :len(entity_id_labels[i + 1]), start_tok_ids.size(0):] # pad entity attentions before combining, pad zeros before for attn 1 and after for attn 2
            ent_seq_attn2 = F.pad(ent_seq_attn2, (doc_seq_len - const.MAX_ENCODER_LENGTH + start_tok_ids.size(0), batch_seq_len - doc_seq_len, len(entity_id_labels[i]), new_entity_len - len(entity_id_labels_combined)))

            ent_seq_attn = (ent_seq_attn1 + ent_seq_attn2)
            ent_seq_attn = ent_seq_attn / (ent_seq_attn.sum(-1, keepdim=True) + 1e-10)

            # -- ENTITY TO ENTITY ATTENTION COMBINATION --
            ent_ent_attn1 = ent_to_ent_attn[i][:, :len(entity_id_labels[i]), :len(entity_id_labels[i])]
            ent_ent_attn1 = F.pad(ent_ent_attn1, (0, new_entity_len - len(entity_id_labels[i]), 0, new_entity_len - len(entity_id_labels[i])))

            ent_ent_attn2 = ent_to_ent_attn[i + 1][:, :len(entity_id_labels[i + 1]), :len(entity_id_labels[i + 1])]
            ent_ent_attn2 = F.pad(ent_ent_attn2, (len(entity_id_labels[i]), new_entity_len - len(entity_id_labels_combined), len(entity_id_labels[i]), new_entity_len - len(entity_id_labels_combined)))

            ent_ent_attn = (ent_ent_attn1 + ent_ent_attn2)
            ent_ent_attn = ent_ent_attn / (ent_ent_attn.sum(-1, keepdim=True) + 1e-10)

            # Append to new tensors
            new_seq_lhs.append(seq_hs)
            new_ent_lhs.append(ent_hs)
            new_ent_to_seq_attn.append(ent_seq_attn)
            new_ent_to_ent_attn.append(ent_ent_attn)
            new_entity_id_labels.append(entity_id_labels_combined)

            i += 2 # Skip to next document (2 because this document was in 2 segments)

    # Pad all entity related tensors to new_entity_len
    for i in range(len(new_entity_id_labels)):
        entity_len_diff = new_entity_len - new_ent_lhs[i].shape[0]
        if entity_len_diff > 0:
            new_ent_lhs[i] = F.pad(new_ent_lhs[i], (0, 0, 0, entity_len_diff))
            new_ent_to_seq_attn[i] = F.pad(new_ent_to_seq_attn[i], (0, 0, 0, entity_len_diff))
            new_ent_to_ent_attn[i] = F.pad(new_ent_to_ent_attn[i], (0, entity_len_diff, 0, entity_len_diff))

    seq_lhs = torch.stack(new_seq_lhs, dim=0)
    ent_lhs = torch.stack(new_ent_lhs, dim=0)
    ent_to_seq_attn = torch.stack(new_ent_to_seq_attn, dim=0)
    ent_to_ent_attn = torch.stack(new_ent_to_ent_attn, dim=0)
    entity_id_labels = new_entity_id_labels

    return seq_lhs, ent_lhs, ent_to_seq_attn, ent_to_ent_attn, entity_id_labels