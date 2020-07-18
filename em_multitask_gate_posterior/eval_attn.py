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
            # for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:
                # print("batch_index", batch_index)
            for input_batch, input_bow_batch, input_length_batch, user_batch, item_batch, target_batch, target_l_batch, target_length_batch, random_flag in eval_data:
                batch_size = input_batch.size(0)

                input_batch_gpu = input_batch.to(self.m_device)

                input_bow_batch_gpu = input_bow_batch.to(self.m_device)

                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_l_batch_gpu = target_l_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch_gpu[:, :-1]
                input_de_length_batch_gpu = target_length_batch_gpu-1

                user_hidden_gpu = self.m_user_embedding(user_batch_gpu)
                item_hidden_gpu = self.m_item_embedding(item_batch_gpu)
                
                max_seq_len = max(target_length_batch-1)
                
                samples, z = self.f_decode_text(user_hidden_gpu, item_hidden_gpu, max_seq_len)

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

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # if batch_index > 1:
                #     break
        
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

    def f_decode_text(self, u, v, max_seq_len, n=4):
        
        u_v = self.m_network.m_latent2embed(u+v)
        z, _ = self.m_network.f_reparameterize_prior(u_v)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        user_item_hidden_i = u_v

        t = 0

        # print("hidden size", hidden.size())
        hidden = None
        
        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
                input_l_seq = torch.zeros(batch_size).fill_(0).long().to(self.m_device)
            
            # input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_x_i = self.m_network.m_embedding(input_seq)
            input_l_i = self.m_network.m_l_embedding(input_l_seq)
            # print("--"*20)
            # print("input_embedding", input_embedding.size())
            # print("var_de", var_de.size())

            output_l_i, hidden_l = self.m_network.m_decoder_l_rnn(torch.cat([input_l_i, input_x_i], dim=-1).unsqueeze(1), hidden_l)

            output_l_i = output_l_i.squeeze(1)
            output_l_logit_i = self.m_network.m_l_output(output_l_i)

            l_i = self._sample(output_l_logit_i)

            input_x_i = input_x_i+l_i.unsqueeze(1)*user_item_hidden_i

            output_i, hidden = self.m_network.m_decoder_rnn(input_x_i.unsqueeze(1), hidden)

            output_i = output_i.squeeze(1)
            output_func_logit_i = self.m_network.m_output2funcvocab(output_i)
            output_func_logit_i[:, self.m_func_vocab_size:] = float('-inf')

            output_logit_i = (1-l_i.unsqueeze(1))*output_func_logit_i

            output_cont_logit_i = self.m_network.m_output2contvocab(output_i)

            output_bow_logit_i = self.m_network.m_bow2contvocab(z)

            output_logit_i = output_logit_i+l_i.unsqueeze(1)*(output_cont_logit_i+output_bow_logit_i)

            x_i = self._sample(output_logit_i) 

            # input_embedding = input_embedding.unsqueeze(1)

            # if torch.isnan(input_embedding).any():
            #     print("input_embedding", input_embedding)

            ## print("input_embedding", input_embedding.size())
            # output, hidden = self.m_network.m_generator.m_decoder_rnn(input_embedding, hidden)

            # logits = self.m_network.m_generator.m_output2vocab(output)

            # input_seq = self._sample(logits)

            # if len(input_seq.size()) < 1:
            #     input_seq = input_seq.unsqueeze(0)

            generations = self._save_sample(generations, x_i, seq_running, t)

            seq_mask[seq_running] = (x_i != self.m_eos_idx).bool()
            seq_running = seq_idx.masked_select(seq_mask)

            running_mask = (x_i != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                x_i = x_i[running_seqs]
                l_i = l_i[running_seqs]

                hidden = hidden[:, running_seqs]
                hidden_l = hidden_l[:, running_seqs]

                user_item_hidden_i = user_item_hidden_i[:, running_seqs]
                # repeat_hidden_0 = repeat_hidden_0[running_seqs]
                output_i = output_i[running_seqs]

                z = z[running_seqs]
                u = u[running_seqs]
                v = v[running_seqs]
                
                u_attn = torch.cat([output_i, u], dim=-1)
                v_attn = torch.cat([output_i, v], dim=-1)

                u_attn_score = self.m_network.m_generator.m_attn(u_attn)
                v_attn_score = self.m_network.m_generator.m_attn(v_attn)

                attn_score = F.softmax(torch.cat([u_attn_score, v_attn_score], dim=-1), dim=-1)
                # print("attn_score", attn_score)
                
                user_item_hidden_i = attn_score[:, 0].unsqueeze(1)*u
                user_item_hidden_i = user_item_hidden_i + attn_score[:, 1].unsqueeze(1)*v

                user_item_hidden_i = self.m_network.m_generator.m_latent2output(user_item_hidden_i)

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)
                                
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