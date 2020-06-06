import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F
import torch.nn as nn
import math

class _EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_i2w = vocab_obj.m_i2w
        
        self.m_embedding_size = args.embedding_size
        self.m_epoch = args.epochs
        self.m_batch_size = args.batch_size 
        self.m_mean_loss = 0

        self.m_x0 = args.x0
        self.m_k = args.k

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
                input_seq = input_seq.unsqueeze(1)

            ### input_seq: batch_size*seq_len

            ### input_embedding: batch_size*seq_len*embedding_size
            input_embedding = self.m_network.m_embedding(input_seq)

            ### input_embedding_attn: seq_len*batch_size*embedding_size
            input_embedding_attn = input_embedding.transpose(0, 1)

            input_embedding_attn = input_embedding_attn*math.sqrt(self.m_embedding_size)
            input_embedding_attn = self.m_network.m_pos_encoder(input_embedding_attn)
            
            ### z_de: batch_size*latent_size
            z_de = self.m_network.m_latent2hidden(z)
            s_de = self.m_network.m_latent2hidden(s)
            l_de = self.m_network.m_latent2hidden(l)

            ### var_de: batch_size*3*latent_size
            var_de = torch.cat([z_de.unsqueeze(1), s_de.unsqueeze(1), l_de.unsqueeze(1)], dim=1)
            
            ### var_de: 3*batch_size*latent_size
            var_de = var_de.transpose(0, 1)
            
            ### output: seq_len*batch_size*latent_size
            output = self.m_network.m_transformer_decoder(input_embedding_attn, var_de, tgt_mask=self.m_network.f_generate_square_subsequent_mask(len(input_embedding_attn)).to(self.m_device))

            ### output: batch_size*seq_len*latent_size
            output = output.transpose(0, 1)

            ### logits: batch_size*latent_size
            logits = self.m_network.m_output2vocab(output[:, -1])

            ### next_word: batch_size*1
            next_word = self._sample(logits)
            
            ### input_seq: batch_size*seq_len
            input_seq = torch.cat([input_seq, next_word], dim=-1)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            next_word = next_word.squeeze()
            generations = self._save_sample(generations, next_word, seq_running, t)

            seq_mask[seq_running] = (next_word != self.m_eos_idx).bool()
            seq_running = seq_idx.masked_select(seq_mask)

            running_mask = (next_word != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)

                z = z[running_seqs]
                s = s[running_seqs]
                l = l[running_seqs]

            t += 1
        # exit()

        return generations, z

    def _sample(self, dist, mode="greedy"):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        # sample = sample.squeeze()

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