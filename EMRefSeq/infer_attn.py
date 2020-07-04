import numpy as np
import torch
import random
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F

class _INFER(object):
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

    def f_inference(self, train_data, eval_data):
        self.m_mean_loss = 0
        # for epoch_i in range(self.m_epoch):
        # batch_size = args.batch_size
        self.m_network.eval()
        infer_loss_list = []
        
        batch_index = 0

        bleu_score_list = []

        output_file = "yelp_edu_eval.txt"
        output_f = open(output_file, "w")
        print("output_file")

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
                # RRe_batch = RRe_batch.to(self.m_device)
                # ARe_batch = ARe_batch.to(self.m_device)

                input_de_batch_gpu = target_batch_gpu[:, :-1]
                input_de_length_batch_gpu = target_length_batch_gpu-1

                logits, z_prior, z_mean, z_logv, z, s_mean, s_logv, s_prior, s, l_mean, l_logv, l, variational_hidden = self.m_network(input_batch_gpu, input_length_batch_gpu, input_de_batch_gpu, input_de_length_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag)       

                # print('random_flag', random_flag)

                # print("*"*10, "encode -->  decode <--", "*"*10)
                
                # print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                if random_flag == 0:
                    mean = z_mean+s_mean+l_mean
                elif random_flag == 1:
                    mean = z_mean+s_mean
                elif random_flag == 2:
                    mean = s_mean+l_mean
                elif random_flag == 3:
                    mean = z_mean+l_mean
                
                max_seq_len = max(target_length_batch-1)
                samples, z, attn_score = self.f_decode_text(z_mean, s_mean, l_mean, max_seq_len)

                target_str = *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx)

                pred_str = *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx)

                for sample_i in range(batch_size):
                    user_i = user_batch[sample_i].item()
                    item_i = item_batch[sample_i].item()

                    target_i = target_str[i]
                    pred_i = pred_str[i]      
                    
                    output_msg = str(user_i)+"\t"+str(item_i)+"\t"+target_i+"\t"+pred_i+"\n"

                    output_f.write(output_msg)

            output_f.close()

            # print("->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            # bleu_score_list.append(bleu_score_batch)
            # print("<-"*10)
            # print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            # print2flag(attn_score)
            # print(*print2flag(user_item_flags), sep='\n')

            # print("<-"*10)
            # print("target", "<-"*10, *idx2word(target_batch[:, 1:,], i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

        # mean_bleu_score = np.mean(bleu_score_list)
        # print("bleu score", mean_bleu_score)

    def f_decode_text(self, z, s, l, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size
        
        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        attn_score = torch.zeros(batch_size, max_seq_len, 3).fill_(self.m_pad_idx).to(self.m_device)

        t = 0

        var_de = self.m_network.m_latent2hidden(z+s+l)

        # print("hidden size", hidden.size())
        hidden = None

        # var_de_flag = torch.zeros(batch_size, 1).to(self.m_device)

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            # input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)

            input_embedding = input_embedding
            input_embedding = input_embedding.unsqueeze(1)

            # print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)

            output = output.squeeze(1)
            z_attn = torch.cat([output, z], dim=-1)
            s_attn = torch.cat([output, s], dim=-1)
            l_attn = torch.cat([output, l], dim=-1)

            # print("z_attn", z_attn)
            # print("s_attn", s_attn)
            # print("l_attn", l_attn)

            z_attn_score_step_i = self.m_network.m_attn(z_attn)
            s_attn_score_step_i = self.m_network.m_attn(s_attn)
            l_attn_score_step_i = self.m_network.m_attn(l_attn)

            attn_score_step_i = F.softmax(torch.cat([z_attn_score_step_i, s_attn_score_step_i, l_attn_score_step_i], dim=-1), dim=-1)

            var_de = attn_score_step_i[:, 0].unsqueeze(1)*z
            var_de = var_de + attn_score_step_i[:, 1].unsqueeze(1)*s
            var_de = var_de + attn_score_step_i[:, 2].unsqueeze(1)*l

            var_de = self.m_network.m_latent2hidden(var_de)
            output = output + var_de
            output = output.unsqueeze(1)

            logits = self.m_network.m_output2vocab(output)

            input_seq = self._sample(logits)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            # print("var_de_flag", var_de_flag, end=" ")
            generations, attn_score = self._save_sample(generations, attn_score, attn_score_step_i, input_seq, seq_running, t)

            seq_mask[seq_running] = (input_seq != self.m_eos_idx).bool()
            seq_running = seq_idx.masked_select(seq_mask)

            running_mask = (input_seq != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                hidden = hidden[:, running_seqs]
                var_de = var_de[running_seqs]
                output = output[running_seqs]

                # repeat_hidden_0 = repeat_hidden_0[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)

                z = z[running_seqs]
                s = s[running_seqs]
                l = l[running_seqs]
                
                # var_de_flag = self.m_network.m_decoder_gate(output.squeeze(1))
                
                # var_de = self.m_network.m_latent2hidden((1-var_de_flag)*z+var_de_flag*s+l)

            t += 1

        return generations, z, attn_score

    def _sample(self, dist, mode="greedy"):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, attn_score, attn_score_step_i, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest
        
        # print("debug 1....", var_de_flag.squeeze().data)
        running_score = attn_score[running_seqs]
        running_score[:, t, :] = attn_score_step_i.data
        attn_score[running_seqs] = running_score

        # print("running_flags", running_flags[:, t])
        # print("debug 2....", user_item_flags)

        return save_to, attn_score

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

def idx2word(idx, i2w, pad_idx):

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