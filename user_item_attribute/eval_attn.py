import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F
import torch.nn as nn
import datetime

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
        # print("eval existing")
        # self.f_eval_rec(train_data, eval_data)

    def f_init_user_item(self, eval_data):
        user2uid = eval_data.m_user2uid
        item2iid = eval_data.m_item2iid

        user_num = len(user2uid)
        item_num = len(item2iid)
        latent_size = self.m_latent_size

        self.m_user_embedding = torch.zeros(user_num, latent_size)
        self.m_item_embedding = torch.zeros(item_num, latent_size)

        self.m_user_num = torch.zeros((user_num, 1))
        self.m_item_num = torch.zeros((item_num, 1))

        self.m_local_embedding = torch.zeros(1, latent_size)
        self.m_local_num = 0

    def f_get_user_item(self, train_data, eval_data):
        s_time = datetime.datetime.now()
        
        self.m_user_embedding = self.m_network.m_user_embedding
        self.m_item_embedding = self.m_network.m_item_embedding

        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)

    def f_eval_new(self, train_data, eval_data):
        # self.f_init_user_item(eval_data)

        self.f_get_user_item(train_data, eval_data)

        batch_index = 0

        precision_list = []
        recall_list = []
        mrr_list = []

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:
                # print("batch_index", batch_index)
                # print("=="*10)
                if batch_index > 0:
                    break

                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                
                user_hidden_gpu = self.m_user_embedding(user_batch_gpu)
                item_hidden_gpu = self.m_item_embedding(item_batch_gpu)
                
                user_item_attr_logits_gpu, mask = self.m_network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
                user_item_attr_logits = user_item_attr_logits_gpu.cpu()

                target_logits = target_batch_gpu.cpu()
                precision_batch, recall_batch = self.f_debug_bow(input_batch, user_item_attr_logits, target_logits)
                # precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_logits)

                if precision_batch == 0 and recall_batch == 0:
                    continue
                
                batch_index += 1

                precision_list.append(precision_batch)
                recall_list.append(recall_batch)

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                if batch_index > 1:
                    break

        print("precision: ", np.mean(precision_list))
        print("recall: ", np.mean(recall_list))

    def f_debug_bow(self, inputs, preds, targets, k=10):
        preds = preds.view(-1, preds.size(1))
        _, indices = torch.topk(preds, k, -1)

        print("--"*10, "inputs", "--"*10)
        for i, i_i in enumerate(inputs):
            for j, i_i_j in enumerate(i_i):
                i_i_j_val = i_i_j.item()
                i_i_j_str = self.m_i2w[str(i_i_j_val)]
                print(i_i_j_str, end=" ")
            print()
            print("--"*10)

        print("--"*10, "targets", "--"*10)
        # a = torch.gather(inputs, 1, indices)
        b = targets
        for i, b_i in enumerate(b):
            for j, b_i_j in enumerate(b_i):
                b_i_j_val = b_i_j.item()
                if b_i_j_val:
                    b_i_j_str = inputs[i, j].item()
                    b_i_j_str = self.m_i2w[str(b_i_j_str)]
                    print(b_i_j_str, end=" ")
            print()
            print("--"*10)

        print("--"*10, "preds", "--"*10)
        a = torch.gather(inputs, 1, indices)
        for _, a_i in enumerate(a):
            for _, a_i_j in enumerate(a_i):
                a_i_j_val = a_i_j.item()
                a_i_j_str = self.m_i2w[str(a_i_j_val)]
                print(a_i_j_str, end=" ")
            print()
            print("--"*10)

        return 1, 1

    def f_eval_bow(self, preds, targets, k=10):
        preds = preds.view(-1, preds.size(1))
        _, indices = torch.topk(preds, k, -1)

        # print("indices", indices)
        # print("targets", targets)

        true_pos = torch.gather(targets, 1, indices)
        true_pos = torch.sum(true_pos, dim=1)
        true_pos = true_pos.float()

        pos = torch.sum(targets, dim=1)
        # print("pos", pos)
        if pos.nonzero().size(0) != len(pos):
            print("error")
            print(pos)
            return 0, 0

        recall = true_pos/pos
        precision = true_pos/k

        recall = torch.mean(recall)
        precision = torch.mean(precision)

        return precision, recall

    def _sample(self, dist, mode="greedy"):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):
        # print(" "*10, "*"*10)
        for word_id in sent:

            if word_id == pad_idx:
                break
            # print('word_id', word_id.item())
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str