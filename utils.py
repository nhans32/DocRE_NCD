import random
import torch
import numpy as np
import json
from tqdm import tqdm



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}



pad_ids = {'input_ids': 1, 'entity_ids': 0, 'entity_position_ids': -1, 'attention_mask': 0, 'entity_attention_mask': 0}
def collate_fn(batch):
    flattened_entities = [[ment for e_ments in x['entity_pos'] for ment in e_ments] for x in batch] # (batch_len, num_entity) mentions

    max_seq_len = max([len(x['input_ids']) for x in batch])
    max_entity_len = max([len(x) for x in flattened_entities])
    max_entity_span_len = max([end - start for e_ments in flattened_entities for start, end in e_ments])

    input_ids = [x['input_ids'] + [pad_ids['input_ids']] * (max_seq_len - len(x['input_ids'])) for x in batch]
    attention_mask = [[1] * len(x['input_ids']) + [pad_ids['attention_mask']] * (max_seq_len - len(x['input_ids'])) for x in batch]

    entity_ids = [[2] * len(e_ments) + [pad_ids['entity_ids']] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_attention_mask = [[1] * len(e_ments) + [pad_ids['entity_attention_mask']] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_position_ids = [[[x for x in range(start, end)] + [pad_ids['entity_position_ids']] * (max_entity_span_len - (end-start)) for start, end in e_ments] + [[pad_ids['entity_position_ids']] * max_entity_span_len] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_id_labels = [[e_i for e_i, e_ments in enumerate(x['entity_pos']) for ment in e_ments] for x in batch]
    
    labels = [x['labels'] for x in batch]
    entity_pos = [x['entity_pos'] for x in batch]
    hts = [x['hts'] for x in batch]

    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'entity_ids': torch.tensor(entity_ids),
        'entity_attention_mask': torch.tensor(entity_attention_mask),
        'entity_position_ids': torch.tensor(entity_position_ids),
        'entity_id_labels': entity_id_labels,
        'labels': labels,
        'entity_pos': entity_pos,
        'hts': hts,
    }



def read_docred(fp, tokenizer, rel2id, max_seq_length=1024):
    # NOTE: Entity positions now are accurate, no need to offset in collate_fn.
    docs = json.load(open(fp))
    samples = []

    for doc in tqdm(docs, desc=fp):
        sents = []
        sent_map = [] # Mapping original token index to wordpiece index
        entities = doc['vertexSet']

        for sent in doc['sents']:
            cur_sent_map = {}
            for tok_i, tok in enumerate(sent):
                tok_wordpiece = tokenizer.tokenize(tok)
                cur_sent_map[tok_i] = len(sents)
                sents.extend(tok_wordpiece)
            cur_sent_map[tok_i + 1] = len(sents)
            sent_map.append(cur_sent_map)
        
        entity_pos = []
        for ent in entities: # Mapping entity word positions to token positions 
            entity_pos.append([])
            for ment in ent:
                start = sent_map[ment['sent_id']][ment['pos'][0]] + 1 # +1 to account for offset of soon to be inserted [CLS] token
                end = sent_map[ment['sent_id']][ment['pos'][1]] + 1
                entity_pos[-1].append((start, end))
        
        pos_pairs = {}
        if 'labels' in doc:
            for label in doc['labels']:
                rel = int(rel2id[label['r']])
                if (label['h'], label['t']) not in pos_pairs:
                    pos_pairs[(label['h'], label['t'])] = []
                pos_pairs[(label['h'], label['t'])].append(rel)

        hts, labels = [], []
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t:
                    hts.append([h, t])

                    if (h, t) in pos_pairs:
                        rel_vect = [0] * len(rel2id)
                        for rel in pos_pairs[(h, t)]:
                            rel_vect[rel] = 1
                    else:
                        rel_vect = [1] + [0] * (len(rel2id) - 1) # NA relation
                    labels.append(rel_vect)
        
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        
        samp = {
            'input_ids': input_ids, # [CLS] + document tokens + [SEP]
            'entity_pos': entity_pos, # List of entities, each entity is a list of mention positions for the entity in the form of (start, end)
            'labels': labels, # List of labels for each entity pair, in the form of a vector 
            'hts': hts, # List of all entity pairs
            'title': doc['title'], # Title of document (for evaluation script)
        }
        samples.append(samp)

    return samples
