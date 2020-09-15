import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn


def get_precision_recall(preds, targets, mask, k=1):

    ### preds: batch_size*tag_num
    preds = preds.view(-1, preds.size(1))

    ### indices: batch_size*k
    _, indices = torch.topk(preds, k, -1)

    # print("targets", targets.size())
    precisoin_list = []
    recall_list = []

    pop_correct_num = 0
    non_pop_correct_num = 0

    for i, pred_index in enumerate(indices):
        pred_i = list(pred_index.numpy())
        target_i = list(targets[i].numpy())
        true_pos = set(target_i) & set(pred_i)
        true_pos_num = len(true_pos)
        
        # print("target", set(target_i))
        # print("pred", set(pred_i))

        # for i, j in enumerate(true_pos):
        #     if j[1] < k:
        #         pop_correct_num += 1
        #     else:
        #         non_pop_correct_num += 1

        precision = true_pos_num/k
        recall = true_pos_num/(sum(mask[i]).item())

        # print(precision, "precision")
        # print(recall, "recall")

        precisoin_list.append(precision)
        recall_list.append(recall)

    precision = np.mean(precisoin_list)
    
    recall = np.mean(recall_list)
    # exit()
    return precision, recall