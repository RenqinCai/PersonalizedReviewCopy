import numpy as np
import torch
import random
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu, get_recall
import torch.nn.functional as F
import datetime

class INFER(object):
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

    def f_get_user_item(self, train_data, eval_data):

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        s_time = datetime.datetime.now()

        self.m_network.eval()
        with torch.no_grad():
            for input_length_batch, input_batch, user_batch, item_batch, target_length_batch, target_batch, random_flag in train_data:

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)
                
                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch
                input_de_length_batch_gpu = target_length_batch

                logits_gpu, z_mean_gpu, z_logv_gpu, z_gpu, s_mean_gpu, s_logv_gpu, s_gpu, ARe_pred_gpu, RRe_pred_gpu = self.m_network(input_batch_gpu, input_length_batch_gpu, input_de_batch_gpu, input_de_length_batch_gpu, user_batch_gpu, random_flag)
                
                logits_gpu, z_mean_gpu, z_logv_gpu, z_gpu, s_mean_gpu, s_logv_gpu, s_gpu, l_mean_gpu, l_logv_gpu, l_gpu, variational_hidden_gpu = self.m_network(input_batch_gpu, input_length_batch_gpu, input_de_batch_gpu, input_de_length_batch_gpu, user_batch_gpu, random_flag)

                z_mean = z_mean_gpu.cpu()
                s_mean = s_mean_gpu.cpu()
                l_mean = l_mean_gpu.cpu()

                for i, user_idx in enumerate(user_batch):
                    user_idx = user_idx.item()
                    if user_idx not in eval_user2uid:
                        continue
                    
                    uid = eval_user2uid[user_idx]
                    # user_num[uid] += 1
                    # uid = user_idx
                    z_mean_i = z_mean[i]
                    self.m_user_embedding[uid] += z_mean_i.detach()
                    self.m_user_num[uid] += 1.0

                    # item_idx = item_batch[i]
                    # iid = item_idx
                    item_idx = item_batch[i].item()
                    if item_idx not in eval_item2iid:
                        continue

                    iid = eval_item2iid[item_idx]
                    # item_num[iid] += 1
                    s_mean_i = s_mean[i]
                    self.m_item_embedding[iid] += s_mean_i.detach()
                    self.m_item_num[iid] += 1.0

                    l_mean_i = l_mean[i]
                    self.m_l_embedding += l_mean_i.detach()
                    self.m_l_num += 1.0

        e_time = datetime.datetime.now()
        print("batch user item duration", e_time-s_time)

        self.m_user_embedding = self.m_user_embedding/self.m_user_num
        self.m_item_embedding = self.m_item_embedding/self.m_item_num

        self.m_l_embedding = self.m_l_embedding/self.m_l_num
            
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

        train_user2uid = train_data.m_user2uid
        train_item2iid = train_data.m_item2iid

        ### initialize 
        self.f_init_user_item(eval_data)

        ### get the user embedding
        self.f_get_user_item(train_data, eval_data)

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        bleu_score_list = []
        self.m_network.eval()
        with torch.no_grad():
            for input_length_batch, input_batch, user_batch, item_batch, target_length_batch, target_batch, random_flag in train_data:

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

                print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                # print("user_batch", user_batch)
                # print("item_batch", item_batch)
                z_mean_gpu = self.m_user_embedding[user_batch].to(self.m_device)
                s_mean_gpu = self.m_item_embedding[item_batch].to(self.m_device)
                l_mean_gpu = self.m_l_embedding.to(self.m_device)

                mean = z_mean_gpu+s_mean_gpu+l_mean_gpu
                # mean = torch.cat([z_mean_gpu, s_mean_gpu], dim=1)
                
                max_seq_len = max(length_batch)
                samples, z = self.f_decode_text(mean, max_seq_len)
            
                print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')


    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = z.size(0)

        init_de_hidden = self.m_network.m_latent2hidden(z)

        # print("init_de_hidden", init_de_hidden[154])
        
        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0

        repeat_hidden_0 = init_de_hidden.unsqueeze(1)

        # print("hidden size", hidden.size())
        hidden = None

        while(t < max_seq_len and len(running_seqs)>0):
            # print("--"*10, "t ", t, "--"*10)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            # print("++"*10, "input_seq", input_seq)
            input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)
            # print("input_embedding", input_embedding[154])

            input_embedding = input_embedding+repeat_hidden_0

            # print("input_embedding plus", input_embedding[154])
            # print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)
            
            output = output.contiguous()
            hidden = hidden.contiguous()
            logits = self.m_network.m_output2vocab(output)

            input_seq = self._sample(logits)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            # print("seq_running", seq_running.size(), seq_running)

            generations = self._save_sample(generations, input_seq, seq_running, t)
            # pred_lens[seq_running] += 1.0

            seq_mask[seq_running] = (input_seq != self.m_eos_idx).bool()

            # print("seq_mask", seq_mask.size(), seq_mask)
            
            seq_running = seq_idx.masked_select(seq_mask)
            
            running_mask = (input_seq != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                hidden = hidden[:, running_seqs]
                repeat_hidden_0 = repeat_hidden_0[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)

            t += 1

        # print("pred_lens", pred_lens)

        return generations, z

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
    print_sent_num = 20
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