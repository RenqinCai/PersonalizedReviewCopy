"""
[user, user_x, item, item_x]
set pooling over attributes
use the user as the context vectors to obtain the attention
"""

from typing import Set
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

        # self.m_attr_linear = nn.Linear(1, self.m_attr_embed_size)
        
        self.m_set_enc = SetTransformer2(self.m_attr_embed_size, 1, self.m_attr_embed_size, dim_hidden=64)
        # self.m_set_enc = SetTransformer2(self.m_attr_embed_size, 1, self.m_attr_embed_size, dim_hidden=64)
        self.m_x_enc = SetTransformer2(64, 1, self.m_attr_embed_size, dim_hidden=64)

        # self.m_set_module = SetTransformer(1, 1, self.m_attr_embed_size, num_inds=16, dim_hidden=64)

        # self.m_x_module = SetTransformer(64, 1, self.m_attr_embed_size, num_inds=16, dim_hidden=64)

        self.m_attr_embedding_user_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_attr_embedding_item_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_output_attr_embedding = nn.Linear(self.m_attr_embed_size*4, self.m_vocab_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        self.m_user_embedding.apply(init_weights)
        self.m_item_embedding.apply(init_weights)

        # self.m_attr_linear.apply(init_weights)

        self.m_output_attr_embedding.apply(init_weights)
        self.m_attr_embedding_user_x.apply(init_weights)
        self.m_attr_embedding_item_x.apply(init_weights)

        # self.m_set_linear.apply(init_weights)
        # self.m_set_context.apply(init_weights)
        
    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask


    def f_get_x(self, attr, attr_lens, item_lens, context, user_item_flag):
    
        if user_item_flag == "user":
            attr = self.m_attr_embedding_user_x(attr)
        
        if user_item_flag == "item":
            attr = self.m_attr_embedding_item_x(attr)

        if user_item_flag == "linear":
            attr = attr.float()
            attr = attr.unsqueeze(-1)
            attr = self.m_attr_linear(attr)

        attr_mask = self.f_generate_mask(attr_lens)
        # attr_mask = ~attr_mask
        attr_set, attr_w = self.m_set_enc(attr, attr_mask)

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
        attr_x, user_w = self.m_x_enc(attr_set_expand, ~item_mask)
        
        return attr_x

    def forward(self, ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, user_ids, item_ids, pos_targets, pos_lens, neg_targets, neg_lens):

        user_embed = self.m_user_embedding(user_ids)                                 
        item_embed = self.m_item_embedding(item_ids)
        
        user_x = self.f_get_x(ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, user_embed, "user")

        item_x = self.f_get_x(ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, item_embed, "item")

        output = torch.cat([user_embed, item_embed, user_x, item_x], dim=-1)

        logits = self.m_output_attr_embedding(output)

        return logits, None, None

    def f_eval_forward(self, ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, user_ids, item_ids):

        ### attr_x_item_user: batch_size*embedding

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        user_x = self.f_get_x(ref_attr_item_user, ref_attr_len_item_user, ref_item_len_user, user_embed, "user")

        item_x = self.f_get_x(ref_attr_user_item, ref_attr_len_user_item, ref_user_len_item, item_embed, "item")

        output = torch.cat([user_embed, item_embed, user_x, item_x], dim=-1)

        logits = self.m_output_attr_embedding(output)

        return logits

def init_weights(m):
    initrange = 0.1

    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -initrange, initrange)
    
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
    
