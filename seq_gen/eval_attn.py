import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os

from torch import nonzero
from metric import get_bleu
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import get_precision_recall_F1

class _EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_i2w = vocab_obj.m_i2w

        print("attr word num", len(self.m_i2w))
        # print(self.m_i2w)

        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_device = device
        self.m_model_path = args.model_path

    def f_init_eval(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        print("eval new")
        self.f_eval_new(train_data, eval_data)

    def f_eval_new(self, train_data, eval_data):

        batch_index = 0

        precision_list = []
        recall_list = []
        F1_list = []

        total_non_one_num = 0
        print('--'*10)
        # print("user output weight", self.m_network.m_user_output.weight)
        # print("item output weight", self.m_network.m_item_output.weight)
        total_num = 0

        self.m_network.eval()
        with torch.no_grad():

            topk = 3

            for input_target_attr_batch, output_target_attr_batch, output_target_attr_len_batch, user_batch, item_batch in eval_data:              
                
                # print("==="*20)

                user_gpu = user_batch.to(self.m_device)

                item_gpu = item_batch.to(self.m_device)

                preds = self.m_network.f_eval_forward(user_gpu, item_gpu)

                precision, recall, F1 = get_precision_recall_F1(preds, output_target_attr_batch, output_target_attr_len_batch, k=topk)

                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)

                total_num += user_batch.size(0)

        print("total_non_one_num", total_non_one_num)
        print("total_num", total_num)

        print("test num", len(precision_list))

        mean_precision = np.mean(precision_list)
        print("precision: ", mean_precision)

        mean_recall = np.mean(recall_list)
        print("recall: ", mean_recall)

        mean_F1 = np.mean(F1_list)
        print("F1: ", mean_F1)

    def f_greedy_decode(self, network, users, items, step_num):
        # step_num = 3

        pre_attrs = None
        batch_size = users.size(0)
        voc_size = network.m_output_attr_embedding_user.weight.data.size(0)

        targets = torch.zeros(batch_size, voc_size).to(users.device)

        for i in range(step_num):
            # print("+++"*10, i, "+++"*10)
            # print(i)
            score = network.f_greedy_decode(users, items, pre_attrs, targets, i)
            
            # print("score", score.size())
            max_attr_index = torch.max(score, dim=1)[1]

            # print(max_attr_index)

            targets[torch.arange(batch_size), max_attr_index] = 1

            if pre_attrs is not None:
                new_pre_attrs = torch.cat([pre_attrs, max_attr_index.unsqueeze(-1)], dim=-1)
            else:
                new_pre_attrs = max_attr_index.unsqueeze(-1)

            pre_attrs = new_pre_attrs

        return pre_attrs

    def f_greedy_decode_conditional(self, network, users, items, cond, step_num):
        # step_num = 3

        # pre_attrs = None
        batch_size = users.size(0)
        voc_size = network.m_output_attr_embedding_user.weight.data.size(0)
        targets = torch.zeros(batch_size, voc_size).to(users.device)
        
        targets[torch.arange(batch_size), cond] = 1
        pre_attrs = cond.unsqueeze(-1)    

        for i in range(1, step_num):
            # print("+++"*10, i, "+++"*10)
            # print(i)
            score = network.f_greedy_decode(users, items, pre_attrs, targets, i)
            
            # print("score", score.size())
            max_attr_index = torch.max(score, dim=1)[1]
            # max_attr_index = torch.randint(voc_size, (batch_size, )).to(users.device)

            # print("max_attr_index", max_attr_index.size())
            # print(max_attr_index)

            targets[torch.arange(batch_size), max_attr_index] = 1

            if pre_attrs is not None:
                new_pre_attrs = torch.cat([pre_attrs, max_attr_index.unsqueeze(-1)], dim=-1)
            else:
                new_pre_attrs = max_attr_index.unsqueeze(-1)

            pre_attrs = new_pre_attrs

        return pre_attrs
