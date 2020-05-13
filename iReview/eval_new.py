import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import datetime

class _EVAL(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        # ### uid: index in user embedding
        # self.m_user2id_map = {}

        # ### iid: index in item embedding
        # self.m_item2id_map = {}

        ###
        # self.m_user_embedding = None
        # self.m_item_embedding = None

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

    def f_get_user_item(self, train_data, eval_data):

        s_time = datetime.datetime.now()

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid
        
        self.m_network.eval()
        with torch.no_grad():
            for input_batch, user_batch, item_batch, target_batch, ARe_batch, RRe_batch, length_batch in train_data:
                input_batch_gpu = input_batch.to(self.m_device)
                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)
                length_batch_gpu = length_batch.to(self.m_device)
                target_batch_gpu = target_batch.to(self.m_device)
                # RRe_batch_gpu = RRe_batch.to(self.m_device)
                # ARe_batch_gpu = ARe_batch.to(self.m_device)

                logits_gpu, z_mean_gpu, z_logv_gpu, z_gpu, s_mean_gpu, s_logv_gpu, s_gpu, ARe_pred_gpu, RRe_pred_gpu = self.m_network(input_batch_gpu, user_batch_gpu, length_batch_gpu)

                z_mean = z_mean_gpu.cpu()
                s_mean = s_mean_gpu.cpu()

                # self.m_user_embedding[user_batch_gpu] = 
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

        self.m_user_embedding = self.m_user_embedding/self.m_user_num
        self.m_item_embedding = self.m_item_embedding/self.m_item_num

        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)
            
    def f_init_user_item(self, eval_data):
        # self.m_user_embedding = self.m_network.m_user_embedding
        # self.m_item_embedding = self.m_network.m_item_embedding

        user2uid = eval_data.m_user2uid
        item2iid = eval_data.m_item2iid

        user_num = len(user2uid)
        item_num = len(item2iid)
        latent_size = self.m_latent_size

        # print('user num', user_num)
        # print('item num', item_num)

        self.m_user_embedding = torch.zeros(user_num, latent_size)
        self.m_item_embedding = torch.zeros(item_num, latent_size)

        self.m_user_num = torch.zeros((user_num, 1))
        self.m_item_num = torch.zeros((item_num, 1))
        
    def f_eval(self, train_data, eval_data):
        self.m_mean_loss = 0
        # for epoch_i in range(self.m_epoch):
        # batch_size = args.batch_size

        infer_loss_list = []
        
        batch_index = 0

        bleu_score_list = []

        ### initialize 
        self.f_init_user_item(eval_data)

        ### get the user embedding
        self.f_get_user_item(train_data, eval_data)

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        ### reconstruct the review

        batch_index = 0

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, user_batch, item_batch, target_batch, ARe_batch, RRe_batch, length_batch in eval_data:

                print("batch index", batch_index)
                batch_index += 1

                input_batch_gpu = input_batch.to(self.m_device)
                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)
                length_batch_gpu = length_batch.to(self.m_device)
                target_batch_gpu = target_batch.to(self.m_device)
                RRe_batch_gpu = RRe_batch.to(self.m_device)
                ARe_batch_gpu = ARe_batch.to(self.m_device)

                # logits, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred = self.m_network(input_batch_gpu, user_batch_gpu, length_batch_gpu)
                # print("*"*10, "encode -->  decode <--", "*"*10)
                # for i, user_idx in enumerate()
                # print("user batch", user_batch.size())
                # print("item batch", item_batch.size())

                z_mean_gpu = self.m_user_embedding[user_batch].to(self.m_device)
                s_mean_gpu = self.m_item_embedding[item_batch].to(self.m_device)

                mean = torch.cat([z_mean_gpu, s_mean_gpu], dim=1)
                max_seq_len = max(length_batch)
                samples, z = self.f_decode_text(mean, max_seq_len)

                lens = length_batch.tolist()
                # lens = pred_lens.cpu().tolist()
                # print("pred_lens", pred_lens)
                # print("lens", lens)
                preds = samples.cpu().tolist()
                target_batch = target_batch.tolist()

                preds = [pred_i[:lens[index]]for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]]for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)
                print("bleu_score_batch", bleu_score_batch)
                bleu_score_list.append(bleu_score_batch)

        mean_bleu_score = np.mean(bleu_score_list)
        print("bleu score", mean_bleu_score)

    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        if len(z.size()) < 2:
            print("size < 2", z.size())
            z = z.unsqueeze(0)
            print("unsqueeze", z.size())

        batch_size = z.size(0)

        init_de_hidden = self.m_network.m_latent2hidden(z)
        
        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        # pred_lens = torch.zeros(batch_size)

        t = 0

        repeat_hidden_0 = init_de_hidden.unsqueeze(1)

        # print("hidden size", hidden.size())
        hidden = None

        while(t < max_seq_len and len(running_seqs)>0):
            # print("--"*10, "t ", t, "--"*10)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq)
            input_embedding = self.m_network.m_embedding(input_seq)

            input_embedding = input_embedding+repeat_hidden_0

            # print("input_embedding", input_embedding.size())
            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)

            output = output.contiguous()
            hidden = hidden.contiguous()
            logits = self.m_network.m_output2vocab(output)

            if (logits == float('inf')).any():
                print("error inf")
                for tmp_i, logits_i in enumerate(logits):
                    print(tmp_i, " -- ", logits_i)

            if (torch.isnan(logits)).any():
                print("error nan")
                for tmp_i, logits_i in enumerate(logits):
                    print(tmp_i, " -- ", logits_i)

            #         print(i, repeat_hidden_0[i])

            input_seq = self._sample(logits)
            # print("input_seq new", input_seq)

            if len(input_seq.size()) < 1:
                input_seq = input_seq.unsqueeze(0)

            generations = self._save_sample(generations, input_seq, seq_running, t)
            # pred_lens[seq_running] = pred_lens[seq_running]+1.0
            
            # print("before seq_r unning", seq_running)
            seq_mask[seq_running] = (input_seq != self.m_eos_idx).bool()
            
            seq_running = seq_idx.masked_select(seq_mask)
            # print("after seq_running", seq_running)

            running_mask = (input_seq != self.m_eos_idx).bool()
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_seq = input_seq[running_seqs]

                hidden = hidden[:, running_seqs]
                repeat_hidden_0 = repeat_hidden_0[running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().to(self.m_device)

            t += 1

        # pred_lens = pred_lens.long()
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