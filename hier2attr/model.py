"""
[user, user_avg_item, item]
"""

import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np
import time

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

        self.m_output_hidden_size = args.output_hidden_size

        self.m_attn_linear_size = args.attn_linear_size

        # self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_attr_embedding_x.weight, -initrange, initrange)

        # torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_avg_attr(self, attr, attr_lens):
        ### attr_user_embed: batch_size*seq_len*embed_size
        attr_embed = self.m_attr_embedding_x(attr) 

        attr_mask = self.f_generate_mask(attr_lens)
        attr_mask = ~attr_mask
        attr_mask = attr_mask.unsqueeze(-1)

        masked_attr_embed = attr_embed*attr_mask

        attr_x = masked_attr_embed.sum(1)/attr_mask.sum(1)

        return attr_x

    def f_get_avg_attr_user(self, attr_item, item_lens):  
        ### attr_user: user_num*item_num*embed_size
        ### item_mask: user_num*item_num

        # print("attr_item", attr_item.size())
        # print("item_mask", item_mask.size())

        # t0 = time.time()

        item_mask = self.f_generate_mask(item_lens)
        item_mask = ~item_mask

        attr_user = torch.zeros((*item_mask.size(), attr_item.size(-1)), device=attr_item.device)

        # print('step 0 {} seconds'.format(time.time() - t0))

        ### item_mask: user_num*item_num
        item_mask = item_mask.bool()
        # item_mask = item_mask.unsqueeze(-1)
        attr_user[item_mask] = attr_item

        attr_user_mean = torch.sum(attr_user, dim=1)
        attr_user = attr_user_mean/torch.sum(item_mask, dim=1, keepdim=True)

        # print('avg user {} seconds'.format(time.time() - t0))
    
        return attr_user

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits
      
    def forward(self, ref_attr_item_user, ref_attr_len_item_user, ref_item_user, ref_item_len_user, user_ids, item_ids, pos_targets, pos_lens, neg_targets, neg_lens):

        attr_x_item_user = self.f_get_avg_attr(ref_attr_item_user, ref_attr_len_item_user)
        attr_x = self.f_get_avg_attr_user(attr_x_item_user, ref_item_len_user)

        user_embed = self.m_user_embedding(user_ids)                                 
        item_embed = self.m_item_embedding(item_ids)
        
        neg_attr_embed_user = self.m_output_attr_embedding_user(neg_targets)
        neg_attr_embed_item = self.m_output_attr_embedding_item(neg_targets)
        neg_attr_embed_x = self.m_attr_embedding_x(neg_targets)

        neg_logits_user = torch.matmul(neg_attr_embed_user, user_embed.unsqueeze(-1))
        neg_logits_user = neg_logits_user.squeeze(-1)

        neg_logits_item = torch.matmul(neg_attr_embed_item, item_embed.unsqueeze(-1))
        neg_logits_item = neg_logits_item.squeeze(-1)

        neg_logits_x = torch.matmul(neg_attr_embed_x, attr_x.unsqueeze(-1))
        neg_logits_x = neg_logits_x.squeeze(-1)

        neg_logits = neg_logits_user+neg_logits_item+neg_logits_x
       
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size

        pos_attr_embed_user = self.m_output_attr_embedding_user(pos_targets)
        pos_attr_embed_item = self.m_output_attr_embedding_item(pos_targets)
        pos_attr_embed_x = self.m_attr_embedding_x(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits_user = torch.matmul(pos_attr_embed_user, user_embed.unsqueeze(-1))
        pos_logits_user = pos_logits_user.squeeze(-1)

        pos_logits_item = torch.matmul(pos_attr_embed_item, item_embed.unsqueeze(-1))
        pos_logits_item = pos_logits_item.squeeze(-1)

        pos_logits_x = torch.matmul(pos_attr_embed_x, attr_x.unsqueeze(-1))
        pos_logits_x = pos_logits_x.squeeze(-1)

        pos_logits = pos_logits_user+pos_logits_item+pos_logits_x

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, ref_attr_item_user, ref_attr_mask_item_user, ref_item_user, ref_item_mask_user, user_ids, item_ids):

        ### attr_x_item_user: batch_size*embedding
        attr_x_item_user = self.f_get_avg_attr(ref_attr_item_user, ref_attr_mask_item_user)

        attr_x = self.f_get_avg_attr_user(attr_x_item_user, ref_item_mask_user)

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        logits_user = torch.matmul(user_embed, self.m_output_attr_embedding_user.weight.t())

        logits_item = torch.matmul(item_embed, self.m_output_attr_embedding_item.weight.t())

        logits_x = torch.matmul(attr_x, self.m_attr_embedding_x.weight.t())

        logits = logits_user+logits_item+logits_x

        return logits