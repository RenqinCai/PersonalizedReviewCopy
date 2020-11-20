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
        
        self.m_set_enc = SET_ENCODER(self.m_attr_embed_size, self.m_attr_embed_size)
        # self.m_set_enc = SetTransformer2(self.m_attr_embed_size, 1, self.m_attr_embed_size, dim_hidden=64)
        self.m_x_enc = SET_DECODER(self.m_attr_embed_size, 1)

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

    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def forward(self, pos_attr_set, pos_attr_lens, neg_attr_set, neg_attr_lens, neg_attr_set_num, user_ids, item_ids, _):
        
        ### neg_attr_sets: (batch_size*neg_attr_set_num)*neg_attr_len

        ### get embedding
        pos_attr_embed_user = self.m_output_attr_embedding_user(pos_attr_set)
        pos_attr_embed_item = self.m_output_attr_embedding_item(pos_attr_set)

        neg_attr_embed_user = self.m_output_attr_embedding_user(neg_attr_set)
        neg_attr_embed_item = self.m_output_attr_embedding_item(neg_attr_set)

        ### user embed as the context vector to output a score
        ### item embed as the context vector to output a score

        user_embed = self.m_user_embedding(user_ids)                                 
        item_embed = self.m_item_embedding(item_ids)

        ### pos score
        ### get set representation 
        pos_attr_mask = self.f_generate_mask(pos_attr_lens)
        
        ### pos_attr_set: batch_size*set_size*embed_size

        pos_attr_set_user = pos_attr_embed_user
        ### user_embed: batch_size*1*embed_size
        # user_embed = user_embed.unsqueeze(-1)
        # pos_user_attr_set_score = self.m_x_enc(pos_attr_set_user, ~pos_attr_mask, user_embed)

        pos_user_attr_set_score = torch.matmul(pos_attr_set_user*(~pos_attr_mask.unsqueeze(-1)), user_embed.unsqueeze(-1))
        pos_user_attr_set_score = torch.sum(pos_user_attr_set_score*(~pos_attr_mask.unsqueeze(-1)), dim=1)
        # print("pos_user_attr_set_score", pos_user_attr_set_score.size())

        pos_attr_set_item = pos_attr_embed_item
        # pos_attr_set_item = self.m_set_enc(pos_attr_embed_item, pos_attr_mask)
        # item_embed = item_embed.unsqueeze(1)
        # pos_item_attr_set_score = self.m_x_enc(pos_attr_set_item, ~pos_attr_mask, item_embed)
        pos_item_attr_set_score = torch.matmul(pos_attr_set_item*(~pos_attr_mask.unsqueeze(-1)), item_embed.unsqueeze(-1))
        pos_item_attr_set_score = torch.sum(pos_item_attr_set_score*(~pos_attr_mask.unsqueeze(-1)), dim=1)

        # print("pos_item_attr_set_score", pos_item_attr_set_score.size())

        ### sum these two 
        pos_attr_set_score = pos_user_attr_set_score + pos_item_attr_set_score

        # print("pos_attr_set_score", pos_attr_set_score.size())

        ### neg score
        ### get set representation 
        neg_attr_mask = self.f_generate_mask(neg_attr_lens)

        # neg_attr_set_user = self.m_set_enc(neg_attr_embed_user, neg_attr_mask)
        neg_attr_set_user = neg_attr_embed_user

        neg_user_embed = user_embed.repeat_interleave(neg_attr_set_num, dim=0)
        neg_user_attr_set_score = torch.matmul(neg_attr_set_user*(~neg_attr_mask.unsqueeze(-1)), neg_user_embed.unsqueeze(-1))
        neg_user_attr_set_score = torch.sum(neg_user_attr_set_score*(~neg_attr_mask.unsqueeze(-1)), dim=1)

        # neg_user_attr_set_score = self.m_x_enc(neg_attr_set_user, ~neg_attr_mask, neg_user_embed)

        # neg_attr_set_item = self.m_set_enc(neg_attr_embed_item, neg_attr_mask)
        neg_attr_set_item = neg_attr_embed_item

        neg_item_embed = item_embed.repeat_interleave(neg_attr_set_num, dim=0)
        neg_item_attr_set_score = torch.matmul(neg_attr_set_item*(~neg_attr_mask.unsqueeze(-1)), neg_item_embed.unsqueeze(-1))
        neg_item_attr_set_score = torch.sum(neg_item_attr_set_score*(~neg_attr_mask.unsqueeze(-1)), dim=1)

        # neg_item_attr_set_score = self.m_x_enc(neg_attr_set_item, ~neg_attr_mask, neg_item_embed)

        neg_attr_set_score = neg_user_attr_set_score + neg_item_attr_set_score

        dup_pos_attr_set_score = pos_attr_set_score.repeat_interleave(neg_attr_set_num, dim=0)

        logits = dup_pos_attr_set_score-neg_attr_set_score

        # print("logits", logits.size())

        return logits, pos_attr_mask

    def f_greedy_decode(self, user_ids, item_ids, pre_attrs, targets, step_index):

        ### targets: batch_size*attr_size, 0, 1
        ### attrs: <= batch_size*3
        
        target_score = torch.zeros_like(targets)
        batch_size = user_ids.size(0)
        targets = targets.bool()
        ### user_embed: batch_size*embed_size
        user_embed = self.m_user_embedding(user_ids) 
        ### attr_user: attr_size*embed_size
        attr_user = self.m_output_attr_embedding_user.weight.data
        # attr_user = attr_user.unsqueeze(0)

        if step_index > 0:
            ### attr_embed_user: batch_size*seq_len*embed_size
            pre_attr_user = self.m_output_attr_embedding_user(pre_attrs)

            user_score = self.f_get_score(batch_size, user_embed, attr_user, targets, pre_attr_user)
        else:

            ### dup_attr_user: batch_size*attr_size*embed_size
            # dup_attr_user = attr_user.repeat(batch_size, 1, 1)

            # user_score = self.m_x_enc.f_eval_forward(dup_attr_user, user_embed)
            attn_weight = torch.matmul(user_embed, attr_user.t())
            user_score = attn_weight
            # attn_weight = torch.matmul(dup_attr_user, user_embed.unsqueeze(-1))

            # user_score = attn_weight.squeeze(-1)
       
        item_embed = self.m_item_embedding(item_ids)
        attr_item = self.m_output_attr_embedding_item.weight.data
        # attr_item = attr_item.unsqueeze(0)
        
        if step_index > 0:
            ### attr_embed_user: batch_size*seq_len*embed_size
            pre_attr_item = self.m_output_attr_embedding_user(pre_attrs)

            item_score = self.f_get_score(batch_size, item_embed, attr_item, targets, pre_attr_item)
        else:
            attn_weight = torch.matmul(item_embed, attr_item.t())
            # attn_weight = torch.matmul(dup_attr_item, item_embed.unsqueeze(-1))
            item_score = attn_weight
        
        # item_score = attn_weight.squeeze(-1)

        ### score: batch_size*attr_size
        score = user_score+item_score

        score = torch.softmax(score, dim=-1)
        
        if step_index > 0:
            
            target_mask = ~targets
            # target_mask = target_mask.reshape(batch_size, -1)
            target_score[target_mask] = score.flatten()
        else:
            target_score = score

        return target_score

    def f_get_score(self, batch_size, user, attr, targets, pre_attr):
        ### dup_attr: batch_size*attr_size*embed_size
        # dup_attr = attr.repeat(batch_size, 1, 1)

        ### next_attr: batch_size*(attr_size-seq_len)*embed_size
        # targets = targets.bool()
        # next_attr = dup_attr[~targets].reshape(batch_size, -1, attr.size(-1))

        ### next_attr: batch_size*(attr_size-seq_len)*1*embed_size

        # next_attr = next_attr.unsqueeze(2)

        ### pre_attr: batch_size*1*seq_len*embed_size --> batch_size*(attr_size-seq_len)*seq_len*embed_size

        # pre_attr = pre_attr.unsqueeze(1)
        # pre_attr = pre_attr.repeat(1, next_attr.size(1), 1, 1)

        ### input_attr: batch_size*(attr_size-seq_len)*(seq_len+1)*embed_size --> next_input_attr: (batch_size*attr_size-seq_len)*(seq_len+1)*embed_size
        # input_attr = torch.cat([next_attr, pre_attr], dim=2)
        # next_input_attr = input_attr.reshape(-1, input_attr.size(2), input_attr.size(3))

        ### next_user: (batch_size*attr_size-seq_len)*embed_size
        # next_user = user.repeat(input_attr.size(1), 1)

        ### user_score: (batch_size*attr_size-seq_len)*1
        # user_score = self.m_x_enc.f_eval_forward(next_input_attr, next_user)

        ### user: batch_size*attr_embed_size, 
        ### input_attr: attr_size*attr_embed_size
        user_score = torch.matmul(user, attr.t())

        user_score = user_score[~targets]
        user_score = user_score.reshape(batch_size, -1)

        # print("user_score", user_score.size())

        return user_score

def init_weights(m):
    initrange = 0.1

    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -initrange, initrange)
    
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -initrange, initrange)
    
