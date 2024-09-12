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
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'labels': labels,
            'hts': hts,
            'title': doc['title']
        }
        samples.append(samp)

        return samples
