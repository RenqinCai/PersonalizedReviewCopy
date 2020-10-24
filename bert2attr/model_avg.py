"""
[user, attr_x, item]
"""

import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

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

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        self.m_output_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*3)

        # self.m_output_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*2)

        # self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
        # torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_output_attr_embedding.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)
        # torch.nn.init.normal_(self.m_tag_item_embedding.weight, 0.0, 0.01)
        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)


    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_avg_attr(self, attr, attr_lens):
        ### attr_user_embed: batch_size*seq_len*embed_size
        attr_embed = self.m_attr_embedding(attr) 
        # attr_user_embed = attr_user_embed.transpose(0, 1)

        attr_mask = self.f_generate_mask(attr_lens)

        masked_attr_embed = attr_embed*((~attr_mask).unsqueeze(-1))

        attr_x = masked_attr_embed.sum(1)/((~attr_mask).sum(1).unsqueeze(-1))

        return attr_x

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def forward(self, attr, attr_inds, attr_tf, attr_feat, attr_lens, attr_lens_user, attr_lens_item, user_ids, item_ids, pos_targets, pos_lens, neg_targets, neg_lens):

        attr_x = self.f_get_avg_attr(attr, attr_lens)

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        # user_output = torch.cat([user_x, user_embed], dim=-1)
        # item_output = torch.cat([item_x, item_embed], dim=-1)

        user_item_output = torch.cat([user_embed, attr_x, item_embed], dim=-1)
        # user_item_output = torch.cat([user_embed, item_embed], dim=-1)

        neg_embed = self.m_output_attr_embedding(neg_targets)

        ## neg_embed: batch_size*neg_num*embed_size
        ## attr_x: batch_size*embed_size
        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        neg_logits = torch.matmul(neg_embed, user_item_output.unsqueeze(-1))
        neg_logits = neg_logits.squeeze(-1)
        # print("neg_lens", neg_lens)
        # exit()
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        # pos_embed = self.m_attr_embedding(pos_targets)

        pos_embed = self.m_output_attr_embedding(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits = torch.matmul(pos_embed, user_item_output.unsqueeze(-1))
        pos_logits = pos_logits.squeeze(-1)

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, attr, attr_inds, attr_tf, attr_feat, attr_lens, attr_lens_user, attr_lens_item, user_ids, item_ids):

        attr_x = self.f_get_avg_attr(attr, attr_lens)

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        # user_output = torch.cat([user_x, user_embed], dim=-1)
        # item_output = torch.cat([item_x, item_embed], dim=-1)

        user_item_output = torch.cat([user_embed, attr_x, item_embed], dim=-1)
        # user_item_output = torch.cat([user_embed, item_embed], dim=-1)

        logits = torch.matmul(user_item_output, self.m_output_attr_embedding.weight.t())

        return logits