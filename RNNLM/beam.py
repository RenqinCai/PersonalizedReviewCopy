import torch

class Beam(object):
    def __init__(self, beam_size, pad_idx, sos_idx, eos_idx, device):

        self.m_b_size = beam_size
        self.m_done = False
        self.m_pad_idx = pad_idx
        self.m_sos_idx = sos_idx
        self.m_eos_idx = eos_idx

        self.m_device = device

        self.m_scores = torch.zeros(self.m_b_size).to(self.m_device)

        self.m_prevKs = []
        self.m_start_idx = self.m_sos_idx
        # self.m_start_idx = 44
        self.m_nextYs = [torch.LongTensor(self.m_b_size).fill_(self.m_start_idx).to(self.m_device)]

        # self.m_nextYs[0][0] = self.m_sos_idx

        self.m_attn = []

    def get_cur_state(self):
        return self.m_nextYs[-1]

    def get_cur_origin(self):
        return self.m_prevKs[-1]

    def advance(self, workd_lk):
        ### workd_lk: K*words

        num_words = workd_lk.size(1)

        if len(self.m_prevKs) > 0:
            beam_lk = workd_lk+self.m_scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.m_b_size, 0, True, True)

        # print("best_scores", best_scores.cpu().tolist(), best_scores_id.cpu().tolist())

        self.m_scores = best_scores

        prev_k = best_scores_id/num_words

        self.m_prevKs.append(prev_k)
        self.m_nextYs.append(best_scores_id-prev_k*num_words)

        if self.m_nextYs[-1][0] == self.m_eos_idx:
            self.m_done = True

        return self.m_done     

    def sort_best(self):
        return torch.sort(self.m_scores, 0, True)
    
    def get_best(self):
        scores, ids = self.sort_best()

        return scores[1], ids[1]

    def get_hyp(self, k):
        hyp = []

        # print('--'*20)
        for j in range(len(self.m_prevKs)-1, -1, -1):
            hyp.append(self.m_nextYs[j+1][k])
            k = self.m_prevKs[j][k]
            # print("k", k)
        
        return hyp[::-1]

class BSD(object):
    def __init__(self, config, model_weights, src, trg, beam_size=1):
        self.m_config = config
        self.m_model_weights = model_weights
        self.m_beam_size = beam_size

        self.m_src = src
        self.m_trg = trg

        self.m_src_dict = src['w2i']
        self.m_tgt_dict = trg['w2i']

        self._load_model()

    def _load_model(self):
        print("loading pretrained model")

        ####

    def get_hidden_representation(self, input):
        src_emb = self.m_model.src_embedding(input)
        h0_encoder, c0_encoder = self.m_model.get_state(src_emb)
        src_h, src_h_t = self.m_model.encoder(src_emb, h0_encoder)

        h_t = src_h_t[-1]

        return src_h, h_t

    def get_init_state_decoder(self, input):
        decoder_init_state = nn.Tanh()(self.m_model.encoder2decoder(input))

        return decoder_init_state

    def decode_batch(self, idx):
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(self.m_src['data'], self.m_src_dict, idx, self.config['data']['batch_size'], self.config['data']['max_src_length'], add_start=True, add_end=True)

        beam_size = self.m_beam_size

        context_h, (context_h_t, context_c_t) = self.get_hidden_representation(input_lines_src)
        
        context_h = context_h.transpose(0, 1)

        batch_size = context_h.size(1)

        context = torch.tensor(context_h.repeat(beam_size))

        dec_states = [torch.tensor(c_t.repeat(beam_size))]

        beam = [Beam(beam_size, self.m_tgt_dict, device) for k in range(batch_size)]

        dec_out = self.get_init_state_decoder(dec_states[0].squeeze(0))

        dec_states[0] = dec_out

        batch_idx = list(range(batch_size))

        remaining_sents = batch_size

        for i in range(self.config['data']['max_trg_length']):
            input = torch.stack([b.get_cur_state() for b in beam if not b.done]).t().contiguous().view(1, -1)

            trg_h, (trg_h_t, trg_c_t) = self.m_model.decoder(trg_emb, (dec_states[0].squeeze(0), dec_states[1].squeeze(0)), context)

            dec_states =(trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            
            out = F.softmax(self.m_model.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()

            active = []

            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:
                    sent_states = dec_state.view(-1, beam_size, remaining_sents, dec_state.size(2))[:, :, idx]

                    sent_states.data.copy_(sent_states.data.index_select(1, beam[b].get_cur_origin()))

            if not active:
                break

            active_idx = torch.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                view = t.data.view(-1, remaining_sents, self.m_model.decoder.hidden_size)

                new_size = list(t.size())
                new_size[-2] = new_size[-2]*len(active_idx) // remaining_sents

            return torch.tensor(view.index_select(1, active_idx).view(*new_size))

            dec_states = (update_active(dec_states[0]), update_active(dec_states[1]))

            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        all_hyp, all_scores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
    