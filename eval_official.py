# FROM: https://github.com/wzhouad/ATLOP/blob/main/evaluation.py
# NOTE: ONLY USED TO EVALUATE MODEL WITH NO HOLDOUTS
import os
import os.path
import json
import torch


def detailed_supervised_evaluate(predictions,
                                 labels,
                                 id2rel_holdout):
    tp, tn, fp, fn = 0, 0, 0, 0
    macro_precision, macro_recall, macro_f1 = 0, 0, 0
    rel_stats = {}

    for relid, rel in id2rel_holdout.items():
        if relid != 0: # Ignore the 'NA' relation
            rel_preds = predictions[:, relid]
            rel_labels = labels[:, relid]

            rel_tp = torch.logical_and((rel_preds == 1), (rel_labels == 1)).sum().item()
            rel_tn = torch.logical_and((rel_preds == 0), (rel_labels == 0)).sum().item()
            rel_fp = torch.logical_and((rel_preds == 1), (rel_labels == 0)).sum().item()
            rel_fn = torch.logical_and((rel_preds == 0), (rel_labels == 1)).sum().item()

            rel_precision = rel_tp / (rel_tp + rel_fp + 1e-8) # 1e-8 to prevent division by zero
            rel_recall = rel_tp / (rel_tp + rel_fn + 1e-8)
            rel_f1 = (2 * rel_precision * rel_recall) / (rel_precision + rel_recall + 1e-8)

            rel_stats[rel] = {
                'precision': rel_precision,
                'recall': rel_recall,
                'f1': rel_f1
            }

            macro_precision += rel_precision
            macro_recall += rel_recall
            macro_f1 += rel_f1

            tp += rel_tp
            tn += rel_tn
            fp += rel_fp
            fn += rel_fn

    macro_precision = macro_precision / (len(id2rel_holdout) - 1) # -1 to ignore the 'NA' relation
    macro_recall = macro_recall / (len(id2rel_holdout) - 1)
    macro_f1 = macro_f1 / (len(id2rel_holdout) - 1)

    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8) 

    stats = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'rel_stats': rel_stats
    }

    return stats



def to_official(preds, 
                samples,
                id2rel):
    h_idx, t_idx, title = [], [], []

    for samp in samples:
        hts = samp["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [samp["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = torch.nonzero(pred).flatten().tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res



def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train



def official_evaluate(official_preds, 
                      data_dir,
                      stats_save_path=None):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(data_dir, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(data_dir, "train_annotated.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(data_dir, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(data_dir, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    official_preds.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [official_preds[0]]
    for i in range(1, len(official_preds)):
        x = official_preds[i]
        y = official_preds[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(official_preds[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    stats = {
        'f1': re_f1,
        'f1_ign': re_f1_ignore_train_annotated
    }

    return stats