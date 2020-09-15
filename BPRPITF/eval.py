import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os

import torch.nn.functional as F
import torch.nn as nn
import datetime
from metric import get_precision_recall


class _EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_i2w = vocab_obj.m_i2w

        print("tag num", len(self.m_i2w))
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
        # print("eval existing")
        # self.f_eval_rec(train_data, eval_data)

    def f_get_user_item(self, train_data, eval_data):
        s_time = datetime.datetime.now()
        
        self.m_tag_user_embedding = self.m_network.m_tag_user_embedding
        self.m_tag_item_embedding = self.m_network.m_tag_item_embedding

        self.m_user_embedding = self.m_network.m_user_embedding
        self.m_item_embedding = self.m_network.m_item_embedding

        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)

    def f_data_analysis(self, train_data, eval_data):
        target_attr_num_list = []
        input_attr_num_list = []

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_freq_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:

                # input_attr_num = torch.sum(input_length_batch, dim=1)
                input_attr_num_list.extend(input_length_batch)

                target_attr_num = torch.sum(target_batch, dim=1)
                target_attr_num_list.extend(target_attr_num)
        
        input_attr_num = np.mean(input_attr_num_list)

        target_attr_num = np.mean(target_attr_num_list)

        print("input_attr_num", input_attr_num, np.var(input_attr_num_list))
        print("target_attr_num", target_attr_num, np.var(target_attr_num_list))

    def f_get_pred(self, network, user_ids, item_ids):
        ### user_x = batch_size*embed_size
        user_x = network.m_user_embedding(user_ids)

        ### item_x = batch_size*embed_size
        item_x = network.m_item_embedding(item_ids)

        ### m_tag_user_embedding: tag_num*embed_size
        ### user_logits: batch_size*tag_num
        user_logits = torch.matmul(user_x, network.m_tag_user_embedding.weight.data.transpose(0, 1))

        ### m_tag_item_embedding: tag_num*embed_size
        ### item_logits: batch_size*tag_num
        item_logits = torch.matmul(item_x, network.m_tag_item_embedding.weight.data.transpose(0, 1))

        ### pred_logits: batch_size*tag_num
        pred_logits = user_logits+item_logits

        return pred_logits

    def f_eval_new(self, train_data, eval_data):
       
        self.f_get_user_item(train_data, eval_data)

        batch_index = 0

        precision_list = []
        recall_list = []
        mrr_list = []

        print('--'*10)
        # print("user output weight", self.m_network.m_user_output.weight)
        # print("item output weight", self.m_network.m_item_output.weight)

        self.m_network.eval()
        with torch.no_grad():
            pop_correct_num_total = 0
            non_pop_correct_num_total = 0
            pred_num_total = 0
            topk = 3

            for pos_tag_batch, mask_batch, user_batch, item_batch in eval_data:    

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                preds = self.f_get_pred(self.m_network, user_batch_gpu, item_batch_gpu)
                precision, recall = get_precision_recall(preds.cpu(), pos_tag_batch, mask_batch, k=topk)

                # precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_logits)

                if precision == 0 and recall == 0:
                    continue
                
                batch_index += 1

                precision_list.append(precision)
                recall_list.append(recall)

        print("precision: ", np.mean(precision_list))
        print("recall: ", np.mean(recall_list))
