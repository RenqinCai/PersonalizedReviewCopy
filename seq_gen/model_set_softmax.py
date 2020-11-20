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

        # self.m_logit_linear = nn.Linear(1, self.m_attr_embed_size)

        self.m_dropout = nn.Dropout(p=0.2)

        self.m_output_attr_embedding_user = nn.Linear(self.m_attr_embed_size, self.m_vocab_size, bias=False)
        self.m_output_attr_embedding_item = nn.Linear(self.m_attr_embed_size, self.m_vocab_size, bias=False)
        
        # self.m_x_enc = SetTransformer2(self.m_attr_embed_size, num_outputs=self.m_vocab_size, dim_output=1, dim_hidden=64, num_heads=1)

        self.m_label_transform_1 = nn.Linear(self.m_vocab_size, 200)

        self.m_label_transform_2 = nn.Linear(200, self.m_vocab_size)

        self.m_activate_fn_1 = nn.Sigmoid()

        # self.m_x_ac_fc = nn.Sigmoid()
        
        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        self.m_user_embedding.apply(init_weights)
        self.m_item_embedding.apply(init_weights)

        self.m_output_attr_embedding_user.apply(init_weights)
        self.m_output_attr_embedding_item.apply(init_weights)

        # self.m_logit_linear.apply(init_weights)

        # self.m_label_transform.apply(init_weights)

        self.m_label_transform_1.apply(init_weights)
        self.m_label_transform_2.apply(init_weights)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, pos_attr_set, pos_attr_lens, neg_attr_set, neg_attr_lens, neg_attr_set_num, user_ids, item_ids, _):
        
        user_embed = self.m_user_embedding(user_ids)                                 
        item_embed = self.m_item_embedding(item_ids)

        mask = self.f_generate_mask(pos_attr_lens)

        user_embed = self.m_dropout(user_embed)
        item_embed = self.m_dropout(item_embed)

        user_logits = self.m_output_attr_embedding_user(user_embed)
        item_logits = self.m_output_attr_embedding_item(item_embed)

        raw_logits = user_logits+item_logits

        # r = self.m_x_ac_fc(raw_logits)
        # print("raw_logits", raw_logits)
        # print("raw_logits topk", torch.topk(raw_logits, k=3, dim=-1))
        # logits_embed = self.m_logit_linear(self.m_x_ac_fc(raw_logits.unsqueeze(-1)))

        # logits_embed = self.m_logit_linear(raw_logits.unsqueeze(-1))

        # print("logits_embed", logits_embed)
        # print(logits_embed.size())

        # logits, w = self.m_x_enc(logits_embed)

        # print("w", w)
        # print(w[0].size())
        # print("topk", torch.topk(w[0], k=3, dim=-1))
        # print(logits.size())

        # logits = F.relu(self.m_label_transform(raw_logits))+raw_logits
        logits = raw_logits

        # logits = self.m_activate_fn_1(raw_logits)
        # logits = self.m_label_transform(logits)

        # logits = self.m_activate_fn_1(raw_logits)
        # logits = self.m_label_transform_1(logits)
        # logits = F.elu(logits)
        # logits = self.m_label_transform_2(logits)
        # logits = logits + raw_logits

        return logits, None, mask

def init_weights(m):
    initrange = 0.1

    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -initrange, initrange)
    
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
    
