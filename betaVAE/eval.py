import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
from metric import _REC_LOSS, _KL_LOSS_STANDARD, _KL_LOSS_CUSTOMIZE
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
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
        self.m_var_num = args.var_num

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

        self.m_Recon_loss_fn = _REC_LOSS(ignore_index=self.m_pad_idx, device=self.m_device, reduction=True)
        self.m_KL_loss_z_fn = _KL_LOSS_STANDARD(self.m_device, reduction=True)

    def f_eval(self, train_data, eval_data):
        self.f_eval_new(train_data, eval_data)
        self.f_eval_rec(eval_data)
        
    def f_eval_new(self, train_data, eval_data):
        self.f_init_user_item(eval_data)

        self.f_get_user_item(train_data, eval_data)

        eval_user2uid = eval_data.m_user2uid
        eval_item2iid = eval_data.m_item2iid

        # print("user num", len(eval_user2uid))
        # print("item num", len(eval_item2iid))

        # exit()
        batch_index = 0
        bleu_score_list = []

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, target_batch, length_batch, user_batch, item_batch in eval_data:

                input_batch_gpu = input_batch.to(self.m_device)
                length_batch_gpu = length_batch.to(self.m_device)
                target_batch_gpu = target_batch.to(self.m_device)

                z_mean_gpu = self.m_user_embedding[user_batch].to(self.m_device)
                s_mean_gpu = self.m_item_embedding[item_batch].to(self.m_device)

                mean = z_mean_gpu+s_mean_gpu
                mean = mean/2

                max_seq_len = max(length_batch)
                samples, z = self.f_decode_text(mean, max_seq_len)

                lens = length_batch.tolist()
                preds = samples.cpu().tolist()
                target_batch = target_batch.tolist()

                preds = [pred_i[:lens[index]]for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]]for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)

                bleu_score_list.append(bleu_score_batch)

        mean_bleu_score = np.mean(bleu_score_list)
        print("new bleu score", mean_bleu_score)

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
            for input_batch, target_batch, length_batch, user_batch, item_batch in train_data:
                input_batch_gpu = input_batch.to(self.m_device)
                length_batch_gpu = length_batch.to(self.m_device)
                target_batch_gpu = target_batch.to(self.m_device)
                
                logp_gpu, z_mean_gpu, z_logv_gpu, z_gpu, _, _ = self.m_network(input_batch_gpu, length_batch_gpu)

                z_mean = z_mean_gpu.cpu()

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
                        z_mean_i = z_mean[i]
                        
                        self.m_item_embedding[iid] += z_mean_i.detach()
                        self.m_item_num[iid] += 1.0
                        
        self.m_user_embedding = self.m_user_embedding/self.m_user_num
        self.m_item_embedding = self.m_item_embedding/self.m_item_num

        if torch.isnan(self.m_user_num).any():
            print('self.m_user_num', self.m_user_num)

        if torch.isnan(self.m_item_num).any():
            print("self.m_item_num", self.m_item_num)
        
        e_time = datetime.datetime.now()
        print("load user item duration", e_time-s_time)

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

    def f_eval_rec(self, eval_data):
        self.f_rec_bleu(eval_data)
        # self.f_nll(eval_data)

    def f_log_prob(self, z, mu, logvar):
        var = torch.exp(logvar)
        logp = -(z-mu)**2/(2*var) - torch.log(2*np.pi*var)/2

        return logp.sum(dim=1)

    def f_nll(self, eval_data):
        self.m_mean_loss = 0

        infer_loss_list = []
        
        hidden_size = self.m_network.m_hidden_size
        batch_size = self.m_batch_size

        batch_index = 0

        total_nll = 0
        total_sents_num = 0
        total_words_num = 0
        total_rec = 0
        total_kl = 0
        total_elbo = 0

        for input_batch, target_batch, length_batch in eval_data:
            input_batch_gpu = input_batch.to(self.m_device)
            length_batch_gpu = length_batch.to(self.m_device)
            target_batch_gpu = target_batch.to(self.m_device)

            total_sents_num += input_batch.size(0)
            total_words_num += torch.sum(length_batch).item()

            en_seq = self.m_network.f_encode(input_batch_gpu, length_batch_gpu)

            z_mean = self.m_network.m_hidden2mean_z(en_seq)
            z_logv = self.m_network.m_hidden2logv_z(en_seq)
            z_std = torch.exp(0.5*z_logv)

            zero_prior = torch.zeros_like(z_mean)

            tmp = []
            tmp_kl = []
            tmp_rec = []
            tmp_elbo = []

            KL_loss_z, KL_weight_z = self.m_KL_loss_z_fn(z_mean, z_logv, 0)

            for _ in range(self.m_var_num):
                pre_z = torch.randn_like(z_std)
                z = pre_z*z_std + z_mean
                
                var_seq = self.m_network.m_latent2hidden(z)
                output = self.m_network.f_decode(var_seq, input_batch_gpu)

                logits = self.m_network.m_output2vocab(output.view(-1, output.size(2)))

                logp = F.log_softmax(logits, dim=-1)
                logp = logp.view(batch_size, -1, logits.size(-1))

                NLL_loss = self.m_Recon_loss_fn(logp, target_batch_gpu, target_batch_gpu)
                NLL_loss = NLL_loss.view(batch_size, -1)
                NLL_loss = NLL_loss.sum(dim=1)
                
                tmp_rec.append(NLL_loss.unsqueeze(-1))
                
                # elbo_loss = -KL_loss_z+NLL_loss
                # tmp_elbo.append(elbo_loss.unsqueeze(-1))

                # tmp_kl.append(-KL_loss_z.unsqueeze(-1))

                posterior_loss = self.f_log_prob(z, z_mean, z_logv)
                prior_loss = self.f_log_prob(z, zero_prior, zero_prior)

                ppl_sample = prior_loss-NLL_loss-posterior_loss
                tmp.append(ppl_sample.unsqueeze(-1))

            ppl_batch = torch.logsumexp(torch.cat(tmp, dim=1), dim=1) - np.log(self.m_var_num)
            # print("ppl_batch", ppl_batch)
            total_nll += ppl_batch.sum().item()

            tmp_rec = torch.cat(tmp_rec, dim=1)
            # rec_batch = torch.mean(tmp_rec, dim=1) - np.log(self.m_var_num)
            rec_batch = torch.mean(tmp_rec, dim=1)
            total_rec += rec_batch.sum().item()

            total_kl += KL_loss_z.sum().item()
            # elbo_batch = torch.logsumexp(torch.cat(tmp_elbo, dim=1), dim=1) - np.log(self.m_var_num)
            # total_elbo += elbo_batch.sum().item()
        
        nll = total_nll/total_sents_num
        ppl = np.exp(-total_nll/total_words_num)

        print("total_words_num", total_words_num)
        print("total_sents_num", total_sents_num)

        total_elbo = total_kl+total_rec

        elbo = total_elbo/total_sents_num
        rec_ll = total_rec/total_sents_num

        kl = total_kl/total_sents_num

        print("perplexity:%.4f"%ppl, " elbo:%.4f"%elbo, " rec_ll:%.4f"%rec_ll, "kl:%.4f"%kl)

        return nll, ppl

    def f_rec_bleu(self, eval_data):
        self.m_mean_loss = 0

        infer_loss_list = []
        
        bleu_score_list = []

        hidden_size = self.m_network.m_hidden_size
        batch_size = self.m_batch_size

        batch_index = 0

        self.m_network.eval()
        with torch.no_grad():
            for input_batch, target_batch, length_batch, _, _ in eval_data:
                
                input_batch_gpu = input_batch.to(self.m_device)
                length_batch_gpu = length_batch.to(self.m_device)
                target_batch_gpu = target_batch.to(self.m_device)
                
                logp, z_mean, z_logv, z, _, _ = self.m_network(input_batch_gpu, length_batch_gpu)

                # hidden = torch.randn([batch_size, hidden_size]).to(self.m_device)
                # mean = hidden
                mean = z_mean

                max_seq_len = max(length_batch)
                samples, z = self.f_decode_text(mean, max_seq_len)

                lens = length_batch.tolist()
                preds = samples.cpu().tolist()
                target_batch = target_batch.tolist()

                preds = [pred_i[:lens[index]]for index, pred_i in enumerate(preds)]
                targets = [target_i[:lens[index]]for index, target_i in enumerate(target_batch)]

                bleu_score_batch = get_bleu(preds, targets)

                bleu_score_list.append(bleu_score_batch)

        mean_bleu_score = np.mean(bleu_score_list)
        print("bleu score", mean_bleu_score)
     
    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        if len(z.size()) < 2:
            print("before z", z.size())
            z = z.unsqueeze(0)
            print("z after", z.size())

        batch_size = z.size(0)

        init_de_hidden = self.m_network.m_latent2hidden(z)

        # hidden = init_de_hidden.unsqueeze(0)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)
        # seq_mask = torch.ones(batch_size).to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0

        # print("hidden size", hidden.size())
        repeat_hidden_0 = init_de_hidden.unsqueeze(1)
        # repeat_hidden_0 = repeat_hidden_0.expand(input_embedding.size(0), input_embedding.size(1), init_de_hidden.size(-1))

        hidden = None
        # hidden = init_de_hidden.unsqueeze(0)

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)

            # print("input_seq", input_seq.size())
            input_seq = input_seq.unsqueeze(1)
            input_embedding = self.m_network.m_embedding(input_seq)

            # if input_embedding.size(0) != repeat_hidden_0.size(0):
            #     print("input_embedding", input_embedding.size())
            #     print("repeat_hidden_0", repeat_hidden_0.size())

            # input_embedding = input_embedding
            input_embedding = input_embedding+repeat_hidden_0
            # input_embedding = torch.cat([input_embedding, repeat_hidden_0], dim=-1)
           
            # assert input_embedding.size(0) == repeat_hidden_0.size(0)

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
                repeat_hidden_0 = repeat_hidden_0[running_seqs]

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