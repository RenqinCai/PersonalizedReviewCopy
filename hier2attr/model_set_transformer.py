"""
[user, user_x, item, item_x]
set pooling over attributes
"""

import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np
import time
from modules import *

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

        self.m_set_module = SetTransformer(1, 1, self.m_attr_embed_size, num_inds=16, dim_hidden=64)

        self.m_x_module = SetTransformer(64, 1, self.m_attr_embed_size, num_inds=16, dim_hidden=64)

        self.m_output_attr_embedding_user_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_item_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        # self.m_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        self.m_user_embedding.apply(init_weights)
        self.m_item_embedding.apply(init_weights)

        self.m_output_attr_embedding_user.apply(init_weights)
        self.m_output_attr_embedding_item.apply(init_weights)
        self.m_output_attr_embedding_user_x.apply(init_weights)
        self.m_output_attr_embedding_item_x.apply(init_weights)

        # self.m_set_linear.apply(init_weights)
        # self.m_set_context.apply(init_weights)
        
    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_avg_attr(self, attr, attr_lens):
        ### attr_user_embed: batch_size*seq_len*embed_size
        # attr_embed = self.m_attr_embedding_x(attr)
        # print("attr", attr)
        attr = attr.unsqueeze(-1).float()
        attr_hidden = self.m_enc(attr)

        attr_mask = self.f_generate_mask(attr_lens)
        attr_mask = ~attr_mask
        attr_mask = attr_mask.unsqueeze(-1)

        masked_attr_embed = attr_hidden*attr_mask

        ### attr_x: batch_size*embed_size
        attr_x = masked_attr_embed.sum(1)/attr_mask.sum(1)

        
        print("attr_x", attr_x)

        return attr_x

    def f_get_avg_attr_user(self, attr_item, item_lens, flag):  
        ### attr_user: user_num*item_num*embed_size
        ### item_mask: user_num*item_num

        # print("attr_item", attr_item.size())
        # print("item_mask", item_mask.size())

        # t0 = time.time()

        ### item_mask: batch_size*item_num
        item_mask = self.f_generate_mask(item_lens)
        item_mask = ~item_mask

        ### attr_item: (batch_size*item_num)*embed_size
        ### attr_user: batch_size*item_num*embed_size
        attr_user = torch.zeros((*item_mask.size(), attr_item.size(-1)), device=attr_item.device)

        ### item_mask: user_num*item_num
        item_mask = item_mask.bool()
        # item_mask = item_mask.unsqueeze(-1)
        attr_user[item_mask] = attr_item

        # print('step 0 {} seconds'.format(time.time() - t0))

        ### attr_item_attn: batch_size*item_num*embed_size
        attr_item_attn = torch.tanh(self.m_set_linear(attr_user))

        ### attr_item_weight: batch_size*item_num*1
        attr_item_weight = torch.tanh(self.m_set_context(attr_item_attn))

        attr_item_weight = attr_item_weight.squeeze(-1)

        ### attr_item_weight: batch_size*item_num
        attr_item_weight = F.softmax(attr_item_weight, dim=1)

        # print("attr_user", attr_user.size())
        print("attr_item_weight", attr_item_weight)

        attr_item_weight = attr_item_weight.unsqueeze(-1)

        attr_user = attr_user*attr_item_weight

        ### attr_user: batch_size*item_num*embed_size
        attr_user_sum = torch.sum(attr_user, dim=1)
        attr_user = attr_user_sum/torch.sum(item_mask, dim=1, keepdim=True)

        # print('avg user {} seconds'.format(time.time() - t0))

        return attr_user, item_mask

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def f_get_x(self, attr, item_lens):
        attr = attr.float()
        attr = attr.unsqueeze(-1)
        attr_set = self.m_set_module(attr)

        item_mask = self.f_generate_mask(item_lens)
        item_mask = ~item_mask

        ### attr_item: (batch_size*item_num)*embed_size
        ### attr_user: batch_size*item_num*embed_size
        attr_set_expand = torch.zeros((*item_mask.size(), attr_set.size(-1)), device=attr_set.device)

        ### item_mask: user_num*item_num
        item_mask = item_mask.bool()
        # item_mask = item_mask.unsqueeze(-1)
        attr_set_expand[item_mask] = attr_set

        # print("attr set", attr_set.size())
        # print("attr_set_expand", attr_set_expand.size())
        attr_x = self.m_x_module(attr_set_expand)

        return attr_x
      
    def forward(self, ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, user_ids, item_ids, pos_targets, pos_lens, neg_targets, neg_lens):

        # attr_x_item_user = self.f_get_avg_attr(ref_attr_item_user, ref_attr_len_item_user)
        # user_x, user_mask = self.f_get_avg_attr_user(attr_x_item_user, ref_item_len_user, "user")

        # attr_x_user_item = self.f_get_avg_attr(ref_attr_user_item, ref_attr_len_user_item)
        # item_x, item_mask = self.f_get_avg_attr_user(attr_x_user_item, ref_user_len_item, "item")

        user_x = self.f_get_x(ref_attr_item_user, ref_item_len_user)

        item_x = self.f_get_x(ref_attr_user_item, ref_user_len_item)

        user_embed = self.m_user_embedding(user_ids)                                 
        item_embed = self.m_item_embedding(item_ids)
        
        neg_attr_embed_user = self.m_output_attr_embedding_user(neg_targets)
        neg_attr_embed_item = self.m_output_attr_embedding_item(neg_targets)

        neg_attr_embed_user_x = self.m_output_attr_embedding_user_x(neg_targets)
        neg_attr_embed_item_x = self.m_output_attr_embedding_item_x(neg_targets)

        neg_logits_user = torch.matmul(neg_attr_embed_user, user_embed.unsqueeze(-1))
        neg_logits_user = neg_logits_user.squeeze(-1)

        neg_logits_item = torch.matmul(neg_attr_embed_item, item_embed.unsqueeze(-1))
        neg_logits_item = neg_logits_item.squeeze(-1)

        neg_logits_user_x = torch.matmul(neg_attr_embed_user_x, user_x.unsqueeze(-1))
        neg_logits_user_x = neg_logits_user_x.squeeze(-1)

        neg_logits_item_x = torch.matmul(neg_attr_embed_item_x, item_x.unsqueeze(-1))
        neg_logits_item_x = neg_logits_item_x.squeeze(-1)
        neg_logits = neg_logits_user+neg_logits_item+neg_logits_user_x+neg_logits_item_x
       
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size

        pos_attr_embed_user = self.m_output_attr_embedding_user(pos_targets)
        pos_attr_embed_item = self.m_output_attr_embedding_item(pos_targets)

        pos_attr_embed_user_x = self.m_output_attr_embedding_user_x(pos_targets)
        pos_attr_embed_item_x = self.m_output_attr_embedding_item_x(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits_user = torch.matmul(pos_attr_embed_user, user_embed.unsqueeze(-1))
        pos_logits_user = pos_logits_user.squeeze(-1)

        pos_logits_item = torch.matmul(pos_attr_embed_item, item_embed.unsqueeze(-1))
        pos_logits_item = pos_logits_item.squeeze(-1)

        pos_logits_user_x = torch.matmul(pos_attr_embed_user_x, user_x.unsqueeze(-1))
        pos_logits_user_x = pos_logits_user_x.squeeze(-1)

        pos_logits_item_x = torch.matmul(pos_attr_embed_item_x, item_x.unsqueeze(-1))
        pos_logits_item_x = pos_logits_item_x.squeeze(-1)

        pos_logits = pos_logits_user+pos_logits_item+pos_logits_user_x+pos_logits_item_x

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, user_ids, item_ids):

        ### attr_x_item_user: batch_size*embedding
        # attr_x_item_user = self.f_get_avg_attr(ref_attr_item_user, ref_attr_len_item_user)
        # user_x, user_mask = self.f_get_avg_attr_user(attr_x_item_user, ref_item_len_user, "user")

        # attr_x_user_item = self.f_get_avg_attr(ref_attr_user_item, ref_attr_len_user_item)
        # item_x, item_mask = self.f_get_avg_attr_user(attr_x_user_item, ref_user_len_item, "item")

        user_x = self.f_get_x(ref_attr_item_user, ref_item_len_user)

        item_x = self.f_get_x(ref_attr_user_item, ref_user_len_item)

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        logits_user = torch.matmul(user_embed, self.m_output_attr_embedding_user.weight.t())

        logits_item = torch.matmul(item_embed, self.m_output_attr_embedding_item.weight.t())

        logits_user_x = torch.matmul(user_x, self.m_output_attr_embedding_user_x.weight.t())

        logits_item_x = torch.matmul(item_x, self.m_output_attr_embedding_item_x.weight.t())

        logits = logits_user+logits_item+logits_user_x+logits_item_x

        return logits

def init_weights(m):
    initrange = 0.1

    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -initrange, initrange)
    
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
    
