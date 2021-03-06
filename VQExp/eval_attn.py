import numpy as np
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

        self.m_epoch = args.epochs
        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_latent_size = args.latent_size
        self.m_anneal_func = args.anneal_func
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
            # for k, v in check_point['model'].items():
            #     print(k)
            network.load_state_dict(check_point['model'])

        print("pad", self.m_pad_idx)
        print("eos", self.m_eos_idx)

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

        # self.m_user_embedding = torch.zeros(user_num, latent_size)
        # self.m_item_embedding = torch.zeros(item_num, latent_size)

        self.m_user_embedding = None
        self.m_item_embedding = None
        self.m_user_cluster_prob = None

        self.m_user_num = torch.zeros((user_num, 1))
        self.m_item_num = torch.zeros((item_num, 1))

        self.m_local_embedding = torch.zeros(1, latent_size)
        self.m_local_num = 0

    def f_get_user_item(self, train_data, eval_data):
        s_time = datetime.datetime.now()
        
        self.m_user_embedding = self.m_network.m_user_embedding
        self.m_item_embedding = self.m_network.m_item_embedding
        self.m_user_cluster_prob = self.m_network.m_user_cluster_prob

        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)

    def f_eval_new(self, train_data, eval_data):
        self.f_init_user_item(eval_data)

        self.f_get_user_item(train_data, eval_data)

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        batch_index = 0
        bleu_score_list = []

        total_target_word_num = 0
        total_review_num = 0

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:
                # print("batch_index", batch_index)

                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch_gpu[:, :-1]
                input_de_length_batch_gpu = target_length_batch_gpu-1

                # user_hidden_gpu = self.m_user_embedding(user_batch_gpu)
                item_hidden_gpu = self.m_item_embedding(item_batch_gpu)

                input_user_cluster_prob_gpu = self.m_user_cluster_prob[user_batch_gpu]

                print("---"*20)
                print("input_user_cluster_prob_gpu 0", input_user_cluster_prob_gpu[0])
                print("input_user_cluster_prob_gpu 1", input_user_cluster_prob_gpu[1])
                
                max_seq_len = max(target_length_batch-1)
                
                samples = self.f_decode_text(input_user_cluster_prob_gpu, item_hidden_gpu, max_seq_len)

                lens = target_length_batch-1
                lens = lens.tolist()
                preds = samples.cpu().tolist()
                target_batch = target_batch[:, 1:].tolist()

                preds = [pred_i[:lens[index]] for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]] for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)

                bleu_score_list.append(bleu_score_batch)

                total_target_word_num += sum(lens)
                total_target_word_num -= batch_size

                total_review_num += batch_size

                batch_index += 1
        
        mean_bleu_score = np.mean(bleu_score_list)
        print("generating new reviews bleu score", mean_bleu_score)
        print("total_target_word_num", total_target_word_num)
        print("total_review_num", total_review_num)

    def f_eval_rec(self, train_data, eval_data):
        self.m_mean_loss = 0
        # for epoch_i in range(self.m_epoch):
        # batch_size = args.batch_size

        infer_loss_list = []
        
        batch_index = 0

        bleu_score_list = []
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:
                
                # if batch_index > 0:
                #     break

                # batch_index += 1
                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch_gpu[:, :-1]
                input_de_length_batch_gpu = target_length_batch_gpu-1

                logits, z_prior, z_mean, z_logv, z, s_prior, s_mean, s_logv, s, l_mean, l_logv, l, variational_hidden = self.m_network(input_batch_gpu, input_length_batch_gpu, input_de_batch_gpu, input_de_length_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag)       

                # mean = torch.cat([z_mean, s_mean], dim=1)
                # mean = z_mean+s_mean+l_mean
                # mean = z_mean+s_mean
                # mean = s_mean+l_mean
                if random_flag == 0:
                    mean = z_mean+s_mean+l_mean
                elif random_flag == 1:
                    mean = z_mean+s_mean
                elif random_flag == 2:
                    mean = s_mean+l_mean
                elif random_flag == 3:
                    mean = z_mean+l_mean

                max_seq_len = max(target_length_batch-1)
                samples, z = self.f_decode_text(z_mean, s_mean, l_mean, max_seq_len)

                lens = target_length_batch-1
                lens = lens.tolist()
                preds = samples.cpu().tolist()
                target_batch = target_batch[:, 1:].tolist()

                preds = [pred_i[:lens[index]]for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]]for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)

                bleu_score_list.append(bleu_score_batch)

        mean_bleu_score = np.mean(bleu_score_list)
        print("bleu score", mean_bleu_score)

    def f_decode_text(self, z_cluster_prob, s, max_seq_len, n=4):
        # if z is None:
        #     assert "z is none"

        batch_size = z_cluster_prob.size(0)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0

        var_de = z_cluster_prob @ self.m_user_embedding.transpose(0, 1)
        var_de = var_de + s
        
        var_de = self.m_network.m_generator.m_latent2output(var_de)

        # print("hidden size", hidden.size())
        hidden = None

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            # input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)
            # print("--"*20)
            # print("input_embedding", input_embedding.size())
            # print("var_de", var_de.size())
            if (var_de != var_de).any():
                print("inf var_de", var_de)

            if torch.isnan(var_de).any():
                print("var_de", var_de)

            input_embedding = input_embedding+var_de

            input_embedding = input_embedding.unsqueeze(1)

            print(input_embedding.size())
            # print("input_seq", input_seq)
            if torch.isnan(input_embedding).any():
                print("input_embedding", input_embedding)

            if (input_embedding != input_embedding).any():
                print("inf input embedding", input_embedding)

            ## print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_generator.m_decoder_rnn(input_embedding, hidden)

            logits = self.m_network.m_generator.m_output2vocab(output)
            # print("logits", logits)

            if torch.isnan(logits).any():
                print("logits", logits)

            if (logits != logits).any():
                print("inf logits", logits)

            input_seq = self._sample(logits)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            generations = self._save_sample(generations, input_seq, seq_running, t)

            seq_mask[seq_running] = (input_seq != self.m_eos_idx).bool()
            seq_running = seq_idx.masked_select(seq_mask)

            running_mask = (input_seq != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                hidden = hidden[:, running_seqs]
                var_de = var_de[running_seqs]
                # repeat_hidden_0 = repeat_hidden_0[running_seqs]
                output = output[running_seqs].squeeze(1)

                s = s[running_seqs]

                z_cluster_prob = z_cluster_prob[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)
                
                user_attn_score = self.m_network.m_generator.m_attn(output) @ self.m_user_embedding

                item_attn = self.m_network.m_generator.m_attn(output) * s
                item_attn_score = torch.sum(item_attn, dim=-1, keepdim=True)

                user_attn_score = user_attn_score*z_cluster_prob
                attn_score = F.softmax(torch.cat([user_attn_score, item_attn_score], dim=-1), dim=-1)
                
                var_de = (attn_score[:, :-1]*z_cluster_prob) @ (self.m_user_embedding.transpose(0, 1))

                # var_de = attn_score[:, 0].unsqueeze(1)*z
                var_de = var_de + attn_score[:, -1].unsqueeze(1)*s

                var_de = self.m_network.m_generator.m_latent2output(var_de)

                # print("var_de", var_de.size())
                
            t += 1

        return generations

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