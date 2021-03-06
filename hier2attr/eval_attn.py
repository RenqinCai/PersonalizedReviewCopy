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

        print('--'*10)
        # print("user output weight", self.m_network.m_user_output.weight)
        # print("item output weight", self.m_network.m_item_output.weight)

        self.m_network.eval()
        with torch.no_grad():

            topk = 3

            for ref_attr_item_batch, ref_attr_len_item_batch, ref_item_len_batch, ref_attr_user_batch, ref_attr_len_user_batch, ref_user_len_batch, user_batch, item_batch, target_batch, target_mask_batch in eval_data:			                
                ref_attr_item_gpu = ref_attr_item_batch.to(self.m_device)
                ref_attr_len_item_gpu = ref_attr_len_item_batch.to(self.m_device)
                ref_item_len_gpu = ref_item_len_batch.to(self.m_device)

                ref_attr_user_gpu = ref_attr_user_batch.to(self.m_device)
                ref_attr_len_user_gpu = ref_attr_len_user_batch.to(self.m_device)
                ref_user_len_gpu = ref_user_len_batch.to(self.m_device)
               
                user_gpu = user_batch.to(self.m_device)
                item_gpu = item_batch.to(self.m_device)

                logits = self.m_network.f_eval_forward(ref_attr_item_gpu, ref_attr_len_item_gpu, ref_item_len_gpu, ref_attr_user_gpu, ref_attr_len_user_gpu, ref_user_len_gpu, user_gpu, item_gpu)
                
                precision, recall, F1= get_precision_recall_F1(logits.cpu(), target_batch, target_mask_batch, k=topk)

                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)

        mean_precision = np.mean(precision_list)
        print("precision: ", mean_precision)

        mean_recall = np.mean(recall_list)
        print("recall: ", mean_recall)

        mean_F1 = np.mean(F1_list)
        print("F1: ", mean_F1)
