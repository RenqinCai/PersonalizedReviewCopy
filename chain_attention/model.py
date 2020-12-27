
import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from beam import Beam

class _ATTR_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(_ATTR_NETWORK, self).__init__()

        self.m_device = device
        
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num

        self.m_attr_embed_size = args.attr_emb_size
        self.m_user_embed_size = args.user_emb_size
        self.m_item_embed_size = args.item_emb_size

        self.m_attn_head_num = args.attn_head_num
        self.m_attn_layer_num = args.attn_layer_num

        self.m_attn_linear_size = args.attn_linear_size

        self.m_attr_user_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_attr_item_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_output_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.01
        # torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        # torch.nn.init.uniform_(self.m_output_attr_embedding_user_x.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_item_x.weight, -initrange, initrange)
        
        torch.nn.init.uniform_(self.m_attr_embedding_x.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_x.weight, -initrange, initrange)

        # torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_attr_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_attr_item_embedding.weight, -initrange, initrange)
        
        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, user_ids, item_ids, attr_ids, attr_len):

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        attr_embed = self.m_attr_embedding_x(attr_ids)

        attr_len = attr_len+2
        attr_mask = self.f_generate_mask(attr_len)

        input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1), attr_embed], dim=1)

        # input_embed = attr_embed

        input_embed = input_embed.transpose(0, 1)
        attr_attn = self.m_attn(input_embed, src_key_padding_mask = attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        attr_attn = attr_attn*(~attr_mask.unsqueeze(-1))

        user_hidden = attr_attn[:, 0]
        item_hidden = attr_attn[:, 1]

        ### voc_size*embed_size
        voc_user_embed = self.m_attr_user_embedding.weight
        voc_item_embed = self.m_attr_item_embedding.weight

        user_logits = torch.matmul(user_hidden, voc_user_embed.t())

        item_logits = torch.matmul(item_hidden, voc_item_embed.t())

        logits = user_logits+item_logits

        return logits

    def f_eval(self, user_ids, item_ids, topk):

        preds = self.f_decode_greedy(user_ids, item_ids, topk)

        # preds = self.f_decode_beam(user_ids, item_ids, topk)

        return preds

    def f_decode_beam(self, user_ids, item_ids, topk):
        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        generations = []

        i = 0
        attr_len = torch.zeros(user_ids.size(0)).long().to(self.m_device)
        attr_len = attr_len+2

        beam_size = 3
        batch_size = user_embed.size(0)
        beam = [Beam(beam_size, self.m_device) for k in range(batch_size)]

        expand_user_embed = user_embed.unsqueeze(1).expand(user_embed.size(0), beam_size, user_embed.size(1)).reshape(-1, user_embed.size(1))
        expand_item_embed = item_embed.unsqueeze(1).expand(item_embed.size(0), beam_size, item_embed.size(1)).reshape(-1, item_embed.size(1))

        # print(expand_user_embed.size(), expand_item_embed.size())

        for i in range(topk):
            if i == 0:
                input_embed = torch.cat([expand_user_embed.unsqueeze(1), expand_item_embed.unsqueeze(1)], dim=1)
            else:
            ### attr_ids: batch_size*beam_size*i
                attr_ids = torch.stack([b.get_cur_state() for b in beam]).contiguous().to(self.m_device)

                # print("attr_ids", attr_ids)

                attr_ids = attr_ids.view(-1, attr_ids.size(-1))
                # print("attr ids", attr_ids.size())

                attr_embed = self.m_attr_embedding_x(attr_ids)
                # print("attr embed", attr_embed.size())

                input_embed = torch.cat([expand_user_embed.unsqueeze(1), expand_item_embed.unsqueeze(1), attr_embed], dim=1)

            input_embed = input_embed.transpose(0, 1)
            attr_attn = self.m_attn(input_embed)
            attr_attn = attr_attn.transpose(0, 1)

            user_hidden = attr_attn[:, 0]
            item_hidden = attr_attn[:, 1]

            voc_user_embed = self.m_attr_user_embedding.weight
            voc_item_embed = self.m_attr_item_embedding.weight

            user_logits = torch.matmul(user_hidden, voc_user_embed.t())

            item_logits = torch.matmul(item_hidden, voc_item_embed.t())

            logits = user_logits+item_logits

            probs = F.log_softmax(logits, dim=-1)

            attr_probs = probs.view(batch_size, beam_size, -1).contiguous()
            # print("attr_probs", attr_probs.size())

            for b in range(batch_size):
                beam[b].advance(attr_probs[b])

                ### append attribute into the list
        
            attr_len = attr_len+1

            # print([b.get_cur_state().size() for b in beam])
            # exit()

        all_scores = []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            all_scores += [scores[:n_best]]
            k = ks[:n_best]
            
            hyps = beam[b].get_hyp(k)
            # hyps = torch.cat(hyps)
            # print("hyps", hyps)
            # hyps = hyps.unsqueeze(0)
            generations.append(hyps)

        # print("generations", generations)
        preds = torch.cat(generations, dim=0)
        # exit()
        return preds

    def f_decode_greedy(self, user_ids, item_ids, topk):
        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        generations = []

        i = 0

        while i < topk:
            
            if i == 0:
                input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)
            else:
                attr_ids = torch.cat(generations, dim=1)
                attr_embed = self.m_attr_embedding_x(attr_ids)

                input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1), attr_embed], dim=1)

            input_embed = input_embed.transpose(0, 1)
            attr_attn = self.m_attn(input_embed)
            attr_attn = attr_attn.transpose(0, 1)

            user_hidden = attr_attn[:, 0]
            item_hidden = attr_attn[:, 1]

            voc_user_embed = self.m_attr_user_embedding.weight
            voc_item_embed = self.m_attr_item_embedding.weight

            user_logits = torch.matmul(user_hidden, voc_user_embed.t())

            item_logits = torch.matmul(item_hidden, voc_item_embed.t())

            logits = user_logits+item_logits

            output = greedy(logits)
            generations.append(output)

            i += 1

        preds = torch.cat(generations, dim=1)

        return preds


def greedy(logits):
    _, sample = torch.topk(logits, 1, dim=-1)
    return sample


