import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F
import torch.nn as nn
import datetime
import json

class _EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_i2w = vocab_obj.m_i2w
        self.m_voc_size = vocab_obj.vocab_size

        print("attr word num", len(self.m_i2w))
        # print(self.m_i2w)

        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_device = device
        self.m_model_path = args.model_path

        with open(os.path.join(args.data_dir, args.user_boa_file), 'r', encoding='utf8') as f:
            self.m_user_boa_dict = json.loads(f.read())

        with open(os.path.join(args.data_dir, args.item_boa_file), 'r', encoding='utf8') as f:
            self.m_item_boa_dict = json.loads(f.read())

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
        self.f_eval_new_user(train_data, eval_data)
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
            for input_batch, input_freq_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:

                # input_attr_num = torch.sum(input_length_batch, dim=1)
                input_attr_num_list.extend(input_length_batch)

                target_attr_num = torch.sum(target_batch, dim=1)
                target_attr_num_list.extend(target_attr_num)
        
        input_attr_num = np.mean(input_attr_num_list)

        target_attr_num = np.mean(target_attr_num_list)

        print("input_attr_num", input_attr_num, np.var(input_attr_num_list))
        print("target_attr_num", target_attr_num, np.var(target_attr_num_list))

    def f_eval_new(self, train_data, eval_data):
        # self.f_init_user_item(eval_data)

        self.f_data_analysis(train_data, eval_data)
        # exit()

        self.f_get_user_item(train_data, eval_data)

        batch_index = 0

        precision_list = []
        recall_list = []
        mrr_list = []

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:
                
                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                
                user_item_attr_logits_gpu, mask = self.m_network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
                user_item_attr_logits = user_item_attr_logits_gpu.cpu()

                # precision_batch, recall_batch = self.f_debug_bow(input_batch, user_item_attr_logits, target_batch, k=1)
                # print("=="*10, batch_index, "=="*10)
                # # print("batch_index", batch_index)
                # print(item_batch)

                # if batch_index > 3:
                #     break
                
                # print("here")
                precision_batch, recall_batch = self.f_eval_bow(input_batch, target_batch, k=10)
                
                # precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_logits)

                if precision_batch == 0 and recall_batch == 0:
                    continue
                
                batch_index += 1

                precision_list.append(precision_batch)
                recall_list.append(recall_batch)

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # if batch_index > 1:
                #     break

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

    def f_eval_new_user(self, train_data, eval_data):
        # self.f_init_user_item(eval_data)

        self.f_data_analysis(train_data, eval_data)
        # exit()

        self.f_get_user_item(train_data, eval_data)

        batch_index = 0

        precision_list = []
        recall_list = []
        mrr_list = []

        pop_correct_num_total = 0
        non_pop_correct_num_total = 0
        pred_num_total = 0
        topk = 3

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_freq_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:
                
                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                
                # user_item_attr_logits_gpu, mask = self.m_network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
                # user_item_attr_logits = user_item_attr_logits_gpu.cpu()

                # precision_batch, recall_batch = self.f_debug_bow(input_batch, user_item_attr_logits, target_batch, k=1)
                # print("=="*10, batch_index, "=="*10)
                # # print("batch_index", batch_index)
                # print(item_batch)

                # if batch_index > 0:
                #     break
                
                # print("here")
                pop_correct_num_batch, non_pop_correct_num_batch, precision_batch, recall_batch = self.f_eval_bow_user(user_batch, item_batch, input_batch, target_batch, k=topk)
                
                # precision_batch, recall_batch = self.f_eval_bow(user_item_attr_logits, target_logits)

                if precision_batch == 0 and recall_batch == 0:
                    continue
                
                batch_index += 1

                precision_list.append(precision_batch)
                recall_list.append(recall_batch)

                pop_correct_num_total += pop_correct_num_batch
                non_pop_correct_num_total += non_pop_correct_num_batch

                pred_num_total += batch_size*topk

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # if batch_index > 1:
                #     break

        pop_correct_ratio = pop_correct_num_total/pred_num_total
        non_pop_correct_ratio = non_pop_correct_num_total/pred_num_total

        print("precision: ", np.mean(precision_list))
        print("recall: ", np.mean(recall_list))
        print("pop_correct_ratio", pop_correct_ratio, "non_pop_correct_ratio", non_pop_correct_ratio, "pred_num_total", pred_num_total, "pop_correct_num_total", pop_correct_num_total, "non_pop_correct_num_total", non_pop_correct_num_total)


    def f_debug_bow_user(self, users, items, inputs, targets, k=10):
        batch_size = inputs.size(0)
        a = torch.zeros(batch_size, self.m_voc_size).float()
        # print(self.m_voc_size)

        preds = torch.zeros_like(inputs)
        print("++"*20)

        user_lambda = 0.1
        for sample_i in range(batch_size):
            
            userid_i = users[sample_i].item()
            itemid_i = items[sample_i].item()

            user_boa_boafreq = self.m_user_boa_dict[str(userid_i)]
            user_boa = user_boa_boafreq[0]
            user_boa_freq = user_boa_boafreq[1]

            item_boa_boafreq = self.m_item_boa_dict[str(itemid_i)]
            item_boa = item_boa_boafreq[0]
            item_boa_freq = item_boa_boafreq[1]

            # print("user_boa", user_boa)
            # a[sample_i, user_boa] = 1
            a[sample_i, user_boa] = user_lambda*torch.tensor(user_boa_freq).float()
            a[sample_i, item_boa] += (1-user_lambda)*torch.tensor(item_boa_freq).float()

            input_i = inputs[sample_i]
            preds[sample_i] = a[sample_i, input_i]
            pred_freq_i = preds[sample_i]

            _, indices = torch.topk(pred_freq_i, k)
            
            correct_pred_i = targets[sample_i, indices]
            # true_pred = (pred_i == target_i)*(target_i==1)
            if torch.sum(correct_pred_i) > 0:
                print("user id", userid_i, itemid_i)
                print("--"*10, "preds", "--"*10)
                for j, pred_i_j in enumerate(indices):
                    if correct_pred_i[j] == 0:
                        continue
                    else:
                        print(input_i[pred_i_j].item(), self.m_i2w[str(input_i[pred_i_j].item())], pred_freq_i[pred_i_j].item(), end=" ")
                print()
                print("--"*10, "target", "--"*10)
                for j, pred_i_j in enumerate(indices):
                    if correct_pred_i[j] == 0:
                        continue
                    else:
                        print(self.m_i2w[str(input_i[pred_i_j].item())], end=" ")
                print()

        _, indices = torch.topk(preds, k, -1)
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

    def f_eval_bow_user(self, users, items, inputs, targets, k=10):
        batch_size = inputs.size(0)
        a = torch.zeros(batch_size, self.m_voc_size).float()
        # print(self.m_voc_size)

        preds = torch.zeros_like(inputs).float()

        user_lambda = 0.15
        # print("user_lambda", user_lambda)
        for sample_i in range(batch_size):
            
            userid_i = users[sample_i].item()
            itemid_i = items[sample_i].item()

            user_boa_boafreq = self.m_user_boa_dict[str(userid_i)]
            user_boa = user_boa_boafreq[0]
            user_boa_freq = user_boa_boafreq[1]

            user_boa_freq_sum = sum(user_boa_freq)
            user_boa_freq = np.array(user_boa_freq)*1.0/user_boa_freq_sum

            item_boa_boafreq = self.m_item_boa_dict[str(itemid_i)]
            item_boa = item_boa_boafreq[0]
            item_boa_freq = item_boa_boafreq[1]

            item_boa_freq_sum = sum(item_boa_freq)
            item_boa_freq = np.array(item_boa_freq)*1.0/item_boa_freq_sum

            # print("user_boa", user_boa)
            # a[sample_i, user_boa] = 1
            """addition"""
            # a[sample_i, user_boa] = user_lambda*torch.tensor(user_boa_freq).float()
            # a[sample_i, item_boa] += (1-user_lambda)*torch.tensor(item_boa_freq).float()
            
            """item frequency"""
            # a[sample_i, item_boa] = torch.tensor(item_boa_freq).float()

            """
            multiplication
            """
            # a[sample_i, user_boa] = torch.tensor(user_boa_freq).float()
            # a[sample_i, item_boa] *= torch.tensor(item_boa_freq).float()

            """
            mask by user, then item frequency
            """
            a[sample_i, user_boa] = 1
            # print("--"*20)
            # print(a[sample_i])

            a[sample_i, item_boa] *= (1+torch.tensor(item_boa_freq).float())
            # print("--"*10)
            # print(a[sample_i])

            input_i = inputs[sample_i]
            # print(a[sample_i, input_i])
            
            preds[sample_i] = a[sample_i, input_i]
            # exit()
            print("preds", preds[sample_i])
            
        
        _, indices = torch.topk(preds, k, -1)

        exit()
        # print(preds.nonzero())
        # if len(preds.nonzero()):
        #     print("non zero")
        # print(indices)
        true_pos = torch.gather(targets, 1, indices)

        topk_threshold = 3
        pop_correct_num = 0
        non_pop_correct_num = 0
        nonzero_index = true_pos.nonzero()
        for i, j in enumerate(nonzero_index):
            if j[1] < topk_threshold:
                pop_correct_num += 1
            else:
                non_pop_correct_num += 1

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

        return pop_correct_num, non_pop_correct_num, precision, recall
        
    def f_eval_bow(self, preds, targets, k=10):
        # indices = torch.randint(0, preds.size(1), (preds.size(0), k))
        # print("indices", indices.size())
        # preds = preds.view(-1, preds.size(1))
        # print("here")
        # _, indices = torch.topk(preds, k, -1)
        indices = torch.arange(k).expand([targets.size(0), k])

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