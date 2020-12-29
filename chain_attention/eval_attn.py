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
from metric import get_precision_recall_F1, get_precision_recall_F1_test

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
        # self.f_eval_train(train_data)
        self.f_eval_test(eval_data)

    def f_eval_train(self, train_data):
        precision_list = []
        recall_list = []
        F1_list = []

        print('--'*10)
        s_time = datetime.datetime.now()

        topk = 3
        self.m_network.eval()
        with torch.no_grad():
            for user_batch, item_batch, attr_batch, attr_len_batch, target_batch, target_len_batch in train_data:
                
                user_gpu = user_batch.to(self.m_device)             

                item_gpu = item_batch.to(self.m_device)

                attr_gpu = attr_batch.to(self.m_device)

                attr_len_gpu = attr_len_batch.to(self.m_device)

                target_gpu = target_batch.to(self.m_device)

                logits = self.m_network(user_gpu, item_gpu, attr_gpu, attr_len_gpu)

                print("user_gpu", user_gpu)
                print("item_gpu", item_gpu)
                print("attr_gpu", attr_gpu)
                print("preds", torch.topk(logits, topk, -1)[1])
                print("target_batch", target_batch)

                precision, recall, F1= get_precision_recall_F1(logits.cpu(), target_batch, k=topk)                
                # print("recall%.4f"%recall, end=", ")
                # loss_list.append(loss.item())
                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)
        
                exit()
        mean_precision = np.mean(precision_list)
        print("precision:%.4f"%mean_precision)

        mean_recall = np.mean(recall_list)
        print("recall:%.4f"%mean_recall)

        mean_F1 = np.mean(F1_list)
        print("F1:%.4f"%mean_F1)


    def f_eval_test(self, eval_data):

        precision_list = []
        recall_list = []
        F1_list = []

        print('--'*10)
        s_time = datetime.datetime.now()

        topk = 3
        self.m_network.eval()
        with torch.no_grad():
            for user_batch, item_batch, target_len_batch, target_batch in eval_data:
                
                user_gpu = user_batch.to(self.m_device)

                item_gpu = item_batch.to(self.m_device)

                target_gpu = target_batch.to(self.m_device)

                preds = self.m_network.f_eval(user_gpu, item_gpu, topk)

                print("user_gpu", user_gpu)
                print("item_gpu", item_gpu)
                print("preds", preds)
                print("target_batch", target_batch)

                precision, recall, F1= get_precision_recall_F1_test(preds.cpu(), target_batch, target_len_batch, k=topk)
                
                # print("recall%.4f"%recall, end=", ")
                # loss_list.append(loss.item())
                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)

                exit()

        mean_precision = np.mean(precision_list)
        print("precision:%.4f"%mean_precision)

        mean_recall = np.mean(recall_list)
        print("recall:%.4f"%mean_recall)

        mean_F1 = np.mean(F1_list)
        print("F1:%.4f"%mean_F1)

        e_time = datetime.datetime.now()

        print("epoch duration", e_time-s_time)
