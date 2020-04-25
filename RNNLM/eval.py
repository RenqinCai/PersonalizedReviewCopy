import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu
import torch.nn.functional as F
from beam import Beam

class EVAL(object):
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
        self.m_beam_size = 3

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

    def f_eval(self, eval_data):
        self.m_mean_loss = 0
        # for epoch_i in range(self.m_epoch):
        # batch_size = args.batch_size

        infer_loss_list = []
        
        bleu_score_list = []

        hidden_size = self.m_network.m_hidden_size
        batch_size = self.m_batch_size

        for input_batch, target_batch, length_batch in eval_data:
            
            input_batch = input_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            
            hidden = torch.randn([batch_size, hidden_size]).to(self.m_device)
            mean = hidden

            # print("mean", mean)

            max_seq_len = max(length_batch)
            # samples, z = self.f_decode_text(mean, max_seq_len)
            # pred = samples.cpu().tolist()

            samples, scores = self.f_decode_text_beam(mean, max_seq_len)

            pred = samples
            target = target_batch.cpu().tolist()
            
            target = [target_i[:length_batch.cpu()[index]]for index, target_i in enumerate(target)]

            bleu_score_batch = get_bleu(pred, target)

            # bleu_score_batch = self.f_get_bleu_score(samples.cpu(), target_batch.cpu(), length_batch.cpu())

            bleu_score_list.append(bleu_score_batch)

        mean_bleu_score = np.mean(bleu_score_list)
        print("bleu score", mean_bleu_score)

    def f_decode_text_beam(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size

        beam_size = self.m_beam_size

        ### hidden size: batch_size*hidden_size
        hidden = z

        ### hidden size: 1*batch_size*hidden_size
        hidden = hidden.unsqueeze(0)

        hidden_size = hidden.size()[2]

        ### hidden_beam: 1*(batch_size*beam_size)*hidden_size
        hidden_beam = hidden.repeat(1, beam_size, 1)

        ### beam: batch_size
        beam = [Beam(beam_size, self.m_pad_idx, self.m_sos_idx, self.m_eos_idx, self.m_device) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(max_seq_len):

            ### input: 1*(beam_size*remain_size)
            input = torch.stack([b.get_cur_state() for b in beam if not b.m_done]).t().contiguous().view(1, -1).to(self.m_device)

            ### input_emb: (remain_size*beam_size)*1*embed_size
            input_emb = self.m_network.m_embedding(input.transpose(1, 0))

            ### output: (remain_size*beam_size)*1*hidden_size
            ### hidden_beam: 1*(remain_size*beam_size)*hidden_size
            output, hidden_beam = self.m_network.m_decoder_rnn(input_emb, hidden_beam.unsqueeze(0))

            ### logits: (remain_size*beam_size)*voc_size           
            logits = self.m_network.m_linear_output(output.squeeze(1))

            ### pred_prob: (remain_size*beam_size)*voc_size  
            pred_prob = F.softmax(logits, dim=-1)

            ### word_lk: remain_size*beam_size*voc_size
            word_lk = pred_prob.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()

            active = []

            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk[idx]):
                    active += [b]

                ### hidden_beam: (remain_size*beam_size)*hidden_size
                ### b_hidden: 1*beam_size*1*hidden_size 
                b_hidden = hidden_beam.view(-1, beam_size, remaining_sents,  hidden_size)[:, :, idx]

                ### b_hidden: 1*beam_size*1*hidden_size
                b_hidden.copy_(b_hidden.index_select(1, beam[b].get_cur_origin()))

            if not active:
                break

            ### index of remaining sentences in last round
            active_idx = torch.LongTensor([batch_idx[k] for k in active]).to(self.m_device)

            ### re-index for remaining sentences
            batch_idx = {beam:idx for idx, beam in enumerate(active)}

            def update_active(t):

                ### t_tmp: beam_size*remain_size*hidden_size
                t_tmp = t.view(-1, remaining_sents, hidden_size)

                ###
                new_size = list(t.size())
                new_size[-2] = new_size[-2]*len(active_idx) // remaining_sents

                ### new_t: beam_size*new_remain_size*hidden_size
                new_t = t_tmp.index_select(1, active_idx)
                ### new_t: beam_size*new_remain_size*hidden_size
                new_t = new_t.view(*new_size)
                new_t = torch.tensor(new_t).to(self.m_device)

                return new_t

            ### hidden_beam: (new_remain_size*beam_size)*hidden_size
            hidden_beam = update_active(hidden_beam)

            remaining_sents = len(active)

        all_hyp, all_scores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]

        return all_hyp, all_scores

    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size
        
        hidden = z
        hidden = hidden.unsqueeze(0)

        seq_idx = torch.arange(0, batch_size).long().to(self.m_device)

        seq_running = torch.arange(0, batch_size).long().to(self.m_device)

        seq_mask = torch.ones(batch_size).bool().to(self.m_device)

        running_seqs = torch.arange(0, batch_size).long().to(self.m_device)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        t = 0
        init_hidden = hidden

        while(t < max_seq_len and len(running_seqs)>0):
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            input_seq = input_seq.unsqueeze(1)

            input_embedding = self.m_network.m_embedding(input_seq)
            # print("input_embedding", input_embedding)
            # exit()

            output, hidden = self.m_network.m_decoder_rnn(input_embedding, hidden)

            logits = self.m_network.m_linear_output(output)

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