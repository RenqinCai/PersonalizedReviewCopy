"""
softmax as loss
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

        # self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        # encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        # self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        # self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_output_attr_embedding_user = nn.Linear( self.m_attr_embed_size, self.m_vocab_size, bias=False)
        self.m_output_attr_embedding_item = nn.Linear(self.m_attr_embed_size, self.m_vocab_size, bias=False)

        self.m_linear_tf = nn.Linear(1, 1)

        self.m_dropout = nn.Dropout(p=0.2)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids, pos_targets, pos_lens, neg_targets, neg_lens):

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        user_embed = self.m_dropout(user_embed)
        item_embed = self.m_dropout(item_embed)

        user_logits = self.m_output_attr_embedding_user(user_embed)
        item_logits = self.m_output_attr_embedding_item(item_embed)

        logits = user_logits+item_logits
        
        # logits += attr_tf_user
        # logits += attr_tf_item

        return logits, None, None

    def f_eval_forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids):

        attr_item_mask = self.f_generate_mask(attr_lens_item)
        # item_x = attr_attn_item
        
        # """ user """  
        attr_user_mask = self.f_generate_mask(attr_lens_user)

        # user_output = user_embed
        # item_output = item_embed
        
        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)
        # user_embed = F.normalize(user_embed, dim=1)

        item_embed = self.m_item_embedding(item_ids)
        # item_embed = F.normalize(item_embed, dim=1)

        user_output = user_embed
        item_output = item_embed

        attr_embed_user = self.m_output_attr_embedding_user.weight
        # attr_embed_user = F.normalize(attr_embed_user, dim=1)

        attr_embed_item = self.m_output_attr_embedding_item.weight
        # attr_embed_item = F.normalize(attr_embed_item, dim=1)

        logits_user = torch.matmul(user_output, attr_embed_user.t())
        logits_item = torch.matmul(item_output, attr_embed_item.t())

        scale_weight = 100

        logits = logits_user+logits_item

        # tmp_logits = logits.gather(1, attr_item)

        # tmp_logits += attr_tf_item*(~attr_item_mask)

        # logits.scatter_(1, attr_item, tmp_logits)

        # tmp_logits = logits.gather(1, attr_user)

        # tmp_logits += attr_tf_user*(~attr_user_mask)
        # logits.scatter_(1, attr_user, tmp_logits)

        return logits