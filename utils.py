import random
import torch
import numpy as np
import json
from tqdm import tqdm
import const
import os

def print_dict(d):
    print(json.dumps(d, indent=2))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}



def collate_fn(batch):
    flattened_entities = [[ment for e_ments in x['entity_pos'] for ment in e_ments] for x in batch] # (batch_len, num_entity) mentions

    max_seq_len = max([len(x['input_ids']) for x in batch])
    max_entity_len = max([len(x) for x in flattened_entities])
    max_entity_span_len = max([end - start for e_ments in flattened_entities for start, end in e_ments])

    input_ids = [x['input_ids'] + [const.PAD_IDS['input_ids']] * (max_seq_len - len(x['input_ids'])) for x in batch]
    attention_mask = [[1] * len(x['input_ids']) + [const.PAD_IDS['attention_mask']] * (max_seq_len - len(x['input_ids'])) for x in batch]

    entity_ids = [[2] * len(e_ments) + [const.PAD_IDS['entity_ids']] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_attention_mask = [[1] * len(e_ments) + [const.PAD_IDS['entity_attention_mask']] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_position_ids = [[[x for x in range(start, end)] + [const.PAD_IDS['entity_position_ids']] * (max_entity_span_len - (end-start)) for start, end in e_ments] + [[const.PAD_IDS['entity_position_ids']] * max_entity_span_len] * (max_entity_len - len(e_ments)) for e_ments in flattened_entities]
    entity_id_labels = [[e_i for e_i, e_ments in enumerate(x['entity_pos']) for ment in e_ments] for x in batch]
    
    labels = [x['labels'] for x in batch]
    labels_original = [x['labels_original'] for x in batch]
    entity_pos = [x['entity_pos'] for x in batch]
    hts = [x['hts'] for x in batch]

    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'entity_ids': torch.tensor(entity_ids),
        'entity_attention_mask': torch.tensor(entity_attention_mask),
        'entity_position_ids': torch.tensor(entity_position_ids),
        'entity_id_labels': entity_id_labels, # For each mention, the entity index/id it belongs to
        'labels': labels,
        'labels_original': labels_original,
        'entity_pos': entity_pos,
        'hts': hts,
    }



def read_docred(fp, tokenizer, rel2id):
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
        
        sents = sents[:const.MAX_DOC_LENGTH - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        samp = {
            'input_ids': input_ids, # [CLS] + document tokens + [SEP]
            'entity_pos': entity_pos, # List of entities, each entity is a list of mention positions for the entity in the form of (start, end)
            'labels': labels if len(labels) > 0 else None, # List of labels for each entity pair, in the form of a vector. Modified by remove_holdouts
            'labels_original': labels if len(labels) > 0 else None, # This is to preserve the original labels for evaluation
            'hts': hts, # List of all entity pairs
            'title': doc['title'], # Title of document (for evaluation script)
        }
        samples.append(samp)

    return samples



# CRITERIA FOR HOLDOUT
# 1. Must have at least 100 examples of the relationship occuring independently (not co-occuring with another relationship -> see 2.) in the training set.
# 2. Relationship must not exclusively co-occur with with another relationship
#   - We use the concept of positive and negative spaces. If it co-occurs with another relationship too much then there will be no unlabeled examples of that relationship when we test against the positive space.
# 3. Relationship must have at least 50 examples in the development set (independent [not co-occuring])
#   - Need to have enough examples to evaluate the model on.
# 4. Will evaluate on models ability to predict the relationship on just independent instances and co-occuring instances.
# - Essentially, our training set will never consider these new relationships with another relationship
def get_holdouts(train_samples,
                 dev_samples, 
                 rel2id, 
                 id2rel,
                 min_train_indiv=100, # Minimum number of instances where the relationship appears individually (without coocurence with another relationship)
                 min_dev_indiv=50):
    
    train_indiv_labels = []
    train_labels = []
    for doc in train_samples:
        for label in doc['labels']:
            label_sum = sum(label)
            if label_sum == 1 and label[0] != 1: # not NA and not coocurence
                train_indiv_labels.append(label)
            train_labels.append(label)
    train_indiv_labels = np.array(train_indiv_labels)
    train_labels = np.array(train_labels)

    dev_indiv_labels = []
    dev_labels = []
    for doc in dev_samples:
        for label in doc['labels']:
            label_sum = sum(label)
            if label_sum == 1 and label[0] != 1: # not NA and not coocurence
                dev_indiv_labels.append(label)
            dev_labels.append(label)
    dev_indiv_labels = np.array(dev_indiv_labels)
    dev_labels = np.array(dev_labels)

    train_cts = train_labels.sum(axis=0)
    dev_cts = dev_labels.sum(axis=0)
    train_indiv_cts = train_indiv_labels.sum(axis=0)
    dev_indiv_cts = dev_indiv_labels.sum(axis=0)


    holdout_candidate_ids = np.intersect1d(np.where(train_indiv_cts >= min_train_indiv)[0], 
                                           np.where(dev_indiv_cts >= min_dev_indiv)[0]) 
    holdout_candidate_rels = [id2rel[i] for i in holdout_candidate_ids]

    holdout_rels = np.random.choice(holdout_candidate_rels, 24, replace=False).tolist() # We want 4 batches of 6 holdout relationships
    holdout_rel_batches = [holdout_rels[i:i+6] for i in range(0, 24, 6)]

    with open(os.path.join('out', 'holdout_info.json'), 'w') as f: # Dump holdout relationship information
        json.dump({
            'holdout_rel_batches': [{rel: {
                'id': rel2id[rel],
                'train_ct': int(train_cts[rel2id[rel]]),
                'train_indiv_ct': int(train_indiv_cts[rel2id[rel]]),
                'dev_ct': int(dev_cts[rel2id[rel]]),
                'dev_indiv_ct': int(dev_indiv_cts[rel2id[rel]])
            } for rel in batch} for batch in holdout_rel_batches]
        }, f, indent=2)

    return holdout_rel_batches



# Remove holdout relationships from samples ['label'] field
# Original labels are preserved in the original labels field
# Also modify rel2id and id2rel to reflect the reindexing
def remove_holdouts(samples, 
                    holdout_rels,
                    rel2id,
                    id2rel):

    if len(holdout_rels) == 0:
        return samples, rel2id, id2rel
    
    rel2id_holdout = {k: v for k, v in rel2id.items() if k not in holdout_rels and v != 0} # create without Na
    rel2id_holdout = {k: i+1 for i, (k, v) in enumerate(rel2id_holdout.items())}
    rel2id_holdout['Na'] = 0 # Readd Na

    id2rel_holdout = {v: k for k, v in rel2id_holdout.items()}

    index_switch = {v: rel2id_holdout[k] if k in rel2id_holdout else None for k, v in rel2id.items()} # index dictionary from original label indices to new label indices due to reindexing done in rel2id_holdout
    
    for doc in samples:
        new_labels = []
        for label in doc['labels']:
            new_label = [0] * len(rel2id_holdout)
            for i, x in enumerate(label):
                if x == 1 and index_switch[i] is not None:
                    new_label[index_switch[i]] = 1
            if sum(new_label) == 0: # if no label is present, then it is a NA relationship
                new_label[0] = 1
            new_labels.append(new_label)
        doc['labels'] = new_labels

    return samples, rel2id_holdout, id2rel_holdout
