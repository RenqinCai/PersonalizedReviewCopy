import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))

    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])

        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])

        stats.append(max([sum((s_ngrams&r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))

    return stats

def bleu(stats):
    if len(list(filter(lambda x:x==0, stats))) > 0:
        return 0

    (c, r) = stats[:2]
    log_bleu_prec = sum([np.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])])/4

    return np.exp(min([0, 1-float(r)/c])+log_bleu_prec)

def get_bleu(hypotheses, reference):
    stats = np.zeros(10)

    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    
    return 100*bleu(stats)

def get_recall(preds, targets, k=10):
    
    batch_size = preds.shape[0]
    voc_size = preds.shape[1]

    print("batch_size", batch_size)
    print("voc_size", voc_size)

    idx = bn.argpartition(-preds, k, axis=1)
    hit = targets[np.arange(batch_size)[:, np.newaxis], idx[:, :k]]

    hit = np.count_nonzero(hit, axis=1)
    hit = np.array(hit)
    
    hit = np.squeeze(hit)

    recall = np.array([min(n, k) for n in np.count_nonzero(targets, axis=1)])

    recall = hit/recall

    return recall

def get_precision_recall_train(preds, targets, mask, k=1):
    max_preds, _ = torch.max(preds, dim=1, keepdim=True)
    exp_preds = torch.exp(preds-max_preds)
    # mask = ~mask
    exp_preds = exp_preds*(mask.float())

    preds = exp_preds/torch.sum(exp_preds, dim=1, keepdim=True)

    preds = preds.view(-1, preds.size(1))
    _, indices = torch.topk(preds, k, -1)

    pos = torch.sum(targets, dim=1)

    # if pos.nonzero().size(0) != len(pos):
    #     return 0, 0

    true_pos = torch.gather(targets, 1, indices)
    true_pos = torch.sum(true_pos, dim=1)
    true_pos = true_pos.float()

    recall = true_pos/(pos+1e-23)
    precision = true_pos/k

    recall = torch.mean(recall)
    precision = torch.mean(precision)

    return precision, recall

def get_precision_recall(preds, targets, mask, k=1):
    preds = preds.view(-1, preds.size(1))
    _, indices = torch.topk(preds, k, -1)

    precision_list = []
    recall_list = []

    for i, pred_index in enumerate(indices):
        pred_i = list(pred_index.numpy())
        target_i = list(targets[i].numpy())
        true_pos = set(target_i) & set(pred_i)
        true_pos_num = len(true_pos)

        precision = true_pos_num/k
        recall = true_pos_num/(sum(mask[i]).item())

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    # print(avg_precision, avg_recall)

    return avg_precision, avg_recall