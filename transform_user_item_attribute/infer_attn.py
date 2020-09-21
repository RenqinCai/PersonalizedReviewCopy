from random import sample
import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
import csv

class _INFER(object):
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

    def f_init_infer(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_infer(self, train_data, eval_data):
        print("infer new")
        self.f_infer_new(train_data, eval_data)
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

    def f_data_analysis(self, train_data, eval_data):
        target_attr_num_list = []
        input_attr_num_list = []

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:

                # input_attr_num = torch.sum(input_length_batch, dim=1)
                input_attr_num_list.extend(input_length_batch)

                target_attr_num = torch.sum(target_batch, dim=1)
                target_attr_num_list.extend(target_attr_num)
        
        input_attr_num = np.mean(input_attr_num_list)

        target_attr_num = np.mean(target_attr_num_list)

        print("input_attr_num", input_attr_num, np.var(input_attr_num_list))
        print("target_attr_num", target_attr_num, np.var(target_attr_num_list))

    def f_infer_new(self, train_data, eval_data):
        # self.f_init_user_item(eval_data)

        self.f_data_analysis(train_data, eval_data)
        # exit()

        self.f_get_user_item(train_data, eval_data)

        batch_index = 0

        precision_list = []
        recall_list = []
        mrr_list = []

        # output_file = "yelp_restaurant_test.csv"
        output_file = "yelp_restaurant_train.csv"
        output_f = open(output_file, "w")
        print("output_file")

        writer = csv.writer(output_f)

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch in train_data:
                
                if batch_index > 1000:
                    break

                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                max_len_batch = torch.max(input_length_batch).item()

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                
                user_item_attr_logits_gpu, mask = self.m_network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
                user_item_attr_logits = user_item_attr_logits_gpu.cpu()

                # precision_batch, recall_batch = self.f_debug_bow(input_batch, user_item_attr_logits, target_batch, k=1)
                # print("=="*10, batch_index, "=="*10)
                # print("batch_index", batch_index)
                # print(item_batch)

                # if batch_index > 3:
                #     break

                indices, precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_batch, k=3)
                
                # precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_logits)

                if precision_batch == 0 and recall_batch == 0:
                    continue
                
                batch_index += 1

                precision_list.append(precision_batch)
                recall_list.append(recall_batch)

                batch_size = user_batch.size(0)
                for sample_i in range(batch_size):
                    user_i = user_batch[sample_i].item()
                    item_i = item_batch[sample_i].item()
                    input_i = input_batch[sample_i]
                    input_length_i = input_length_batch[sample_i].item()

                    input_index_i = []
                    input_str_i = []

                    for j, input_i_j in enumerate(input_i):
                        if input_i_j != self.m_pad_idx:
                            input_index_i_j = input_i_j.item()
                            input_index_i.append(str(input_index_i_j))
                            input_str_i.append(self.m_i2w[str(input_index_i_j)])

                    input_index_i = " ".join(input_index_i)
                    input_str_i = " ".join(input_str_i)

                    target_index_i = []
                    target_str_i = []
                    target_i = target_batch[sample_i]
                    for j, target_i_j in enumerate(target_i):
                        if target_i_j:
                            target_index_i_j = input_i[j].item()
                            target_str_i_j = self.m_i2w[str(target_index_i_j)]
                            target_str_i.append(target_str_i_j)
                            target_index_i.append(str(target_index_i_j))

                    index_i = indices[sample_i]
                    pred_index_i = []
                    pred_str_i = []
                    for j, index_i_j in enumerate(index_i):
                        pred_i_j = input_i[index_i_j].item()
                        pred_str_i_j = self.m_i2w[str(pred_i_j)]
                        pred_str_i.append(pred_str_i_j)
                        pred_index_i.append(str(pred_i_j))

                    pred_str_i = " ".join(pred_str_i)
                    target_str_i = " ".join(target_str_i)

                    pred_index_i = " ".join(pred_index_i)
                    target_index_i = " ".join(target_index_i)

                    writer.writerow([str(user_i), str(item_i), str(input_length_i), str(max_len_batch), input_index_i, input_str_i, target_index_i, target_str_i, pred_index_i, pred_str_i])

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # if batch_index > 1:
                #     break
        output_f.close()
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
        # indices = torch.randint(0, preds.size(1), (preds.size(0), k))
        # print("indices", indices.size())
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
            # print("error")
            # print(pos)
            return 0, 0, 0

        recall = true_pos/pos
        precision = true_pos/k

        recall = torch.mean(recall)
        precision = torch.mean(precision)

        return indices, precision, recall

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