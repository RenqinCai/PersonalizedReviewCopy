import numpy as np
import torch
import random
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu, get_recall
import torch.nn.functional as F
import datetime
import csv

class _INFER(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_user_embedding = None
        self.m_item_embedding = None

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_i2w = vocab_obj.m_i2w

        self.m_epoch = args.epochs
        self.m_batch_size = args.batch_size
        self.m_latent_size = args.latent_size
        self.m_mean_loss = 0

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func
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

    def f_get_user_item(self):

        self.m_user_embedding = self.m_network.m_user_embedding
        self.m_item_embedding = self.m_network.m_item_embedding

        self.m_misc_embedding = self.m_network.m_generator.m_misc_embedding

        # self.m_l_embedding = self.m_l_embedding/self.m_l_num
            
    def f_init_user_item(self, eval_data):
        # self.m_user_embedding = self.m_network.m_user_embedding.weight
        # self.m_item_embedding = self.m_network.m_item_embedding.weight
        user2uid = eval_data.m_user2uid
        item2iid = eval_data.m_item2iid

        user_num = len(user2uid)
        item_num = len(item2iid)
      
        latent_size = self.m_latent_size

        self.m_user_embedding = torch.zeros(user_num, latent_size)
        self.m_item_embedding = torch.zeros(item_num, latent_size)

        self.m_user_num = torch.zeros((user_num, 1))
        self.m_item_num = torch.zeros((item_num, 1))

        self.m_l_embedding = torch.zeros(1, latent_size)
        self.m_l_num = 0

    def f_inference(self, train_data, eval_data):
        self.m_mean_loss = 0

        infer_loss_list = []
        
        batch_index = 0

        # train_user2uid = train_data.m_user2uid
        # train_item2iid = train_data.m_item2iid

        ### initialize 
        # self.f_init_user_item(eval_data)

        ### get the user embedding
        self.f_get_user_item()

        # eval_user2uid = eval_data.m_user2uid
        # eval_item2iid = eval_data.m_item2iid

        bleu_score_list = []
        output_file = "1.csv"
        output_f = open(output_file, "w")
        print("output_file")

        writer = csv.writer(output_f)

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:

                if batch_index > 0:
                    break

                batch_index += 1

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)
                
                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch
                input_de_length_batch_gpu = target_length_batch

                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("user_batch", user_batch)
                # print("item_batch", item_batch)
                user_hidden_gpu = self.m_user_embedding(user_batch_gpu)
                item_hidden_gpu = self.m_item_embedding(item_batch_gpu)

                
                max_seq_len = max(target_length_batch-1)
                samples = self.f_decode_text(self.m_misc_embedding, user_hidden_gpu, item_hidden_gpu, max_seq_len)

                # print2flag(attn_score)
            
                # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                target_str = idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx)
                # print(len(target_str))

                pred_str = idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx)

                batch_size = user_batch.size(0)
                for sample_i in range(batch_size):
                    user_i = user_batch[sample_i].item()
                    item_i = item_batch[sample_i].item()

                    target_i = target_str[sample_i]
                    pred_i = pred_str[sample_i]      
                    
                    # output_msg = str(user_i)+"\t"+str(item_i)+"\t"+target_i+" xxxxx "+pred_i+"\n"
                    writer.writerow([str(user_i), str(item_i), target_i, pred_i])

                    # output_f.write(output_msg)

            output_f.close()

    def print2flag(idx):
        print_sent_num = 10
        sent_str = [str()]*print_sent_num

        for sent_i, sent in enumerate(idx):
            if sent_i >= print_sent_num:
                break
            
            for flag_step_i in sent:
                sent_str[sent_i] += str(flag_step_i[0].item()) + str(flag_step_i[1].item()) + str(flag_step_i[2].item())+" "
                print("%.4f:%.4f:%.4f"%(flag_step_i[0].item(), flag_step_i[1].item(), flag_step_i[2].item()), end=" | ")
            print("\n")
        return sent_str
        
    def f_decode_text(self, m, z, s, max_seq_len, n=4):
        m = m.expand(z.size())

        batch_size = z.size(0)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0

        # var_de = self.m_network.m_generator.m_latent2output(m+z+s)
        var_de = self.m_network.m_generator.m_latent2output(z+s)

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
            input_embedding = input_embedding+var_de

            input_embedding = input_embedding.unsqueeze(1)

            if torch.isnan(input_embedding).any():
                print("input_embedding", input_embedding)

            ## print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_generator.m_decoder_rnn(input_embedding, hidden)

            logits = self.m_network.m_generator.m_output2vocab(output)

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

                z = z[running_seqs]
                s = s[running_seqs]
                m = m[running_seqs]
                
                m_z_s = torch.cat([m.unsqueeze(1), s.unsqueeze(1), (z+s).unsqueeze(1)], dim=1)

                gate_flag_step_i = self.m_network.m_generator.m_de_gate(output)
                print("--", gate_flag_step_i)
                gate_flag_step_i = self.m_network.m_generator.f_gumbel_softmax(gate_flag_step_i)

                print("gate_flag_step_i", gate_flag_step_i)

                var_de = torch.sum(m_z_s*gate_flag_step_i.unsqueeze(-1), dim=1)

                # z_attn = torch.cat([output, z], dim=-1)
                # s_attn = torch.cat([output, s], dim=-1)

                # z_attn_score = self.m_network.m_generator.m_attn(z_attn)
                # s_attn_score = self.m_network.m_generator.m_attn(s_attn)

                # attn_score = F.softmax(torch.cat([z_attn_score, s_attn_score], dim=-1), dim=-1)
                # # print("attn_score", attn_score)
                
                # var_de = attn_score[:, 0].unsqueeze(1)*z
                # var_de = var_de + attn_score[:, 1].unsqueeze(1)*s

                var_de = self.m_network.m_generator.m_latent2output(var_de)

                # print("var_de", var_de.size())
                
            t += 1

        return generations

    def _sample(self, dist, mode="greedy"):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        # print("sample", sample)
        # print("dist", dist)
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

    # print_sent_num = len(idx)
    print_sent_num = len(idx)
    sent_str = [str()]*print_sent_num
    # sent_str = [str()]*len(idx)
    # print(i2w)
    for i, sent in enumerate(idx):
        if i >= print_sent_num:
            break
            
        for word_id in sent:

            if word_id == pad_idx:
                break
            # print('word_id', word_id.item())
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str

def print2flag(idx):

    print_sent_num = 10
    sent_str = [str()]*print_sent_num

    for sent_i, sent in enumerate(idx):
        if sent_i >= print_sent_num:
            break
        
        for flag_step_i in sent:
            sent_str[sent_i] += str(flag_step_i[0].item()) + str(flag_step_i[1].item()) + " "
            print("%.4f:%.4f"%(flag_step_i[0].item(), flag_step_i[1].item()), end=" | ")
        print("\n")
    return sent_str
