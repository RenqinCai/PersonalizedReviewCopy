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
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        self.f_eval_new(train_data, eval_data)
        self.f_eval_rec(train_data, eval_data)

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
        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in train_data:

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch_gpu[:, :-1]
                input_de_length_batch_gpu = target_length_batch_gpu-1

                logits_gpu, z_prior_gpu, z_mean_gpu, z_logv_gpu, z_gpu, s_prior_gpu, s_mean_gpu, s_logv_gpu, s_gpu, l_mean_gpu, l_logv_gpu, l_gpu, variational_hidden_gpu = self.m_network(input_batch_gpu, input_length_batch_gpu, input_de_batch_gpu, input_de_length_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag)

                z_mean = z_mean_gpu.cpu()
                s_mean = s_mean_gpu.cpu()
                l_mean = l_mean_gpu.cpu()

                for i, user_idx in enumerate(user_batch):
                    user_idx = user_idx.item()
                    if user_idx in eval_user2uid:
                        uid = eval_user2uid[user_idx]
                        z_mean_i = z_mean[i]

                        self.m_user_embedding[uid] += z_mean_i.detach()
                        self.m_user_num[uid] += 1.0

                    item_idx = item_batch[i].item()
                    if item_idx in eval_item2iid:
                        iid = eval_item2iid[item_idx]
                        s_mean_i = s_mean[i]
                        self.m_item_embedding[iid] += s_mean_i.detach()
                        self.m_item_num[iid] += 1.0

                # self.m_local_embedding += torch.mean(l_mean.detach(), dim=0).unsqueeze(0)
                self.m_local_embedding += torch.mean(l_mean.detach(), dim=0)

                self.m_local_num += 1

        self.m_user_embedding = self.m_user_embedding/self.m_user_num
        self.m_item_embedding = self.m_item_embedding/self.m_item_num
        
        if torch.isnan(self.m_user_num).any():
            print("self.m_user_num", self.m_user_num)

        if torch.isnan(self.m_item_num).any():
            print("self.m_item_num", self.m_item_num)

        print("self.m_local_num", self.m_local_num)
        # print("local size", self.m_local_embedding.size(), self.m_local_embedding)
        self.m_local_embedding = self.m_local_embedding/self.m_local_num

        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)

    def f_eval_new(self, train_data, eval_data):
        self.f_init_user_item(eval_data)

        self.f_get_user_item(train_data, eval_data)

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        batch_index = 0
        bleu_score_list = []

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

                z_mean_gpu = self.m_user_embedding[user_batch].to(self.m_device)
                s_mean_gpu = self.m_item_embedding[item_batch].to(self.m_device)
                # l_mean_gpu = self.m_local_embedding.to(self.m_device)
                l_mean_gpu = torch.cat(batch_size*[self.m_local_embedding]).to(self.m_device)

                mean = z_mean_gpu+s_mean_gpu+l_mean_gpu
                max_seq_len = max(target_length_batch-1)
                
                # print("z_mean_gpu", z_mean_gpu)
                # if torch.isnan(z_mean_gpu).any():
                # print("z size", z_mean_gpu.size())
                # print("s size", s_mean_gpu.size())
                # print("l size", l_mean_gpu.size())
                
                samples, z = self.f_decode_text(z_mean_gpu, s_mean_gpu, l_mean_gpu, max_seq_len)

                lens = target_length_batch-1
                lens = lens.tolist()
                preds = samples.cpu().tolist()
                target_batch = target_batch[:, 1:].tolist()

                preds = [pred_i[:lens[index]] for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]] for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)

                bleu_score_list.append(bleu_score_batch)

                batch_index += 1
        
        mean_bleu_score = np.mean(bleu_score_list)
        print("bleu score", mean_bleu_score)

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

    def f_decode_text(self, z, s, l, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0

        var_de = self.m_network.m_latent2hidden(z+s+l)

        # print("hidden size", hidden.size())
        hidden = None

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            # input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)
            
            input_embedding = input_embedding+var_de

            input_embedding = input_embedding.unsqueeze(1)

            if torch.isnan(input_embedding).any():
                print("input_embedding", input_embedding)

            # print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)

            logits = self.m_network.m_output2vocab(output)

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

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)
        
                # z = z[running_seqs]
                # s = s[running_seqs]
                # l = l[running_seqs]
                
                # z_attn = torch.cat([output, z], dim=-1)
                # s_attn = torch.cat([output, s], dim=-1)
                # l_attn = torch.cat([output, l], dim=-1)

                # z_attn_score = self.m_network.m_attn(z_attn)
                # s_attn_score = self.m_network.m_attn(s_attn)
                # l_attn_score = self.m_network.m_attn(l_attn)

                # attn_score = F.softmax(torch.cat([z_attn_score, s_attn_score, l_attn_score], dim=-1), dim=-1)
                # # print("attn_score", attn_score)
                
                # var_de = attn_score[:, 0].unsqueeze(1)*z
                # var_de = var_de + attn_score[:, 1].unsqueeze(1)*s
                # var_de = var_de + attn_score[:, 2].unsqueeze(1)*l

                # var_de = self.m_network.m_latent2hidden(var_de)

                # var_de_flag = self.m_network.m_decoder_gate(output.squeeze(1))
                # var_de = self.m_network.m_latent2hidden((1-var_de_flag)*z+var_de_flag*s+l)

            t += 1

        return generations, z

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