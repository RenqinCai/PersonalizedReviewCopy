import numpy as np
import torch
import random
from nltk.translate.bleu_score import sentence_bleu
import os
from metric import get_bleu, get_recall
import torch.nn.functional as F

class INFER(object):
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

    def f_inference(self, eval_data):
        self.m_mean_loss = 0
        # for epoch_i in range(self.m_epoch):
        # batch_size = args.batch_size

        infer_loss_list = []
        
        batch_index = 0

        bleu_score_list = []
        with torch.no_grad():
            for input_batch, user_batch,  target_batch, ARe_batch, RRe_batch, length_batch in eval_data:

                if batch_index > 0:
                    break

                batch_index += 1

                input_batch = input_batch.to(self.m_device)
                user_batch = user_batch.to(self.m_device)
                length_batch = length_batch.to(self.m_device)                
                ARe_batch = ARe_batch.to(self.m_device)

                logits, s_mean, s_logv, s = self.m_network(input_batch, user_batch, length_batch)
                # print("*"*10, "encode -->  decode <--", "*"*10)
                recall = get_recall(logits.cpu().numpy(), ARe_batch.cpu().numpy())
                print("recall", np.mean(recall))

                print("encoding", "->"*10, *idx2word(input_batch, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

                mean = logits
                # print("encoder_hidden", init_de_hidden)
                
                samples, scores = self.f_decode_BOW(mean, length_batch)

                # recall = get_recall(samples.cpu().numpy(), target_batch.cpu().numpy())
                # print("recall", np.mean(recall))
                
                print("decoding", "<-"*10, *idx2word(samples, i2w=self.m_i2w, pad_idx=self.m_pad_idx), sep='\n')

            # mean_bleu_score = np.mean(bleu_score_list)
            # print("bleu score", mean_bleu_score)
            
    def f_eval(self, pred, target, length):

        pred = pred.tolist()
        pred = [pred_i[:length[index]]for index, pred_i in enumerate(pred)]
        target = target.tolist()
        target = [target_i[:length[index]]for index, target_i in enumerate(target)]

        bleu_score = get_bleu(pred, target)
        return bleu_score

    def f_decode_BOW(self, z, length_batch, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size

        # hidden = self.m_network.m_latent2hidden(z)
        # init_hidden = self.m_network.m_hidden2hidden(z)

        # word_logits = self.m_network.m_output2vocab(z)

        word_probs = F.log_softmax(z, dim=-1)
        print("word_probs", word_probs.size())

        max_seq_len = max(length_batch)

        generations = torch.zeros(batch_size, max_seq_len).fill_(self.m_pad_idx).to(self.m_device).long()

        for sample_index in range(batch_size):
            length_index = length_batch[sample_index].item()
            # sampled_words_index = np.random.choice(self.m_vocab_size, length_index, word_probs[sample_index].cpu().tolist())
            topk_probs, sampled_words_index = torch.topk(word_probs[sample_index], length_index)

            print("topk_probs", topk_probs, sampled_words_index)

            generations[sample_index, :length_index] = sampled_words_index
            # generations[sample_index, :length_index] = torch.from_numpy(sampled_words_index).to(self.m_device)
        
        return generations, z

    def f_decode_text(self, z, max_seq_len, n=4):
        if z is None:
            assert "z is none"

        batch_size = self.m_batch_size

        init_de_hidden = self.m_network.m_latent2hidden(z)
        
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
            # print("t", t)
            if t == 0:
                input_seq = torch.zeros(batch_size).fill_(self.m_sos_idx).long().to(self.m_device)
            
            input_seq = input_seq.unsqueeze(1)
            # print("input seq size", input_seq.size())
            input_embedding = self.m_network.m_embedding(input_seq)

            input_embedding = input_embedding+repeat_hidden_0

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

    print_sent_num = 10
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