"""
model_avg+attribute_pop
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

        self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

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

    def f_get_avg_attr_user(self, attr, attr_lens):
        ### attr_user_embed: batch_size*seq_len*embed_size

        attr_user_embed = self.m_attr_embedding(attr)

        attr_user_mask = self.f_generate_mask(attr_lens)

        masked_attr_user_embed = attr_user_embed*((~attr_user_mask).unsqueeze(-1))

        attr_user = masked_attr_user_embed.sum(1)/((~attr_user_mask).sum(1).unsqueeze(-1))

        return attr_user, attr_user_mask

    def f_get_avg_attr_item(self, attr, attr_lens):
        
        attr_item_embed = self.m_attr_embedding(attr)

        attr_item_mask = self.f_generate_mask(attr_lens)

        masked_attr_item_embed = attr_item_embed*((~attr_item_mask).unsqueeze(-1))
        
        attr_item = masked_attr_item_embed.sum(1)/((~attr_item_mask).sum(1).unsqueeze(-1))

        return attr_item, attr_item_mask

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids, pos_targets, pos_lens, neg_targets, neg_lens):
        # print("==="*10)

        """ item """

        # attr_attn_item, attr_item_mask = self.f_get_avg_attr_item(attr_item, attr_lens_item)   
       
        attr_item_mask = self.f_generate_mask(attr_lens_item)
        # item_x = attr_attn_item
        
        # """ user """  
        attr_user_mask = self.f_generate_mask(attr_lens_user)

        # attr_attn_user, attr_user_mask = self.f_get_avg_attr_user(attr_user, attr_lens_user)
        
        # user_x = attr_attn_user

        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)
        # user_embed = F.normalize(user_embed, dim=1)
        user_embed = self.m_dropout(user_embed)

        item_embed = self.m_item_embedding(item_ids)
        # item_embed = F.normalize(item_embed, dim=1)
        item_embed = self.m_dropout(item_embed)

        user_output = user_embed
        item_output = item_embed
        
        neg_embed_user = self.m_output_attr_embedding_user(neg_targets)
        # neg_embed_user = F.normalize(neg_embed_user, dim=1)
        neg_embed_item = self.m_output_attr_embedding_item(neg_targets)
        # neg_embed_item = F.normalize(neg_embed_item, dim=1)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num

        neg_logits_user = self.f_get_logits(neg_embed_user, user_output)
        neg_logits_item = self.f_get_logits(neg_embed_item, item_output)

        # print("neg_lens", neg_lens)
        # exit()
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num

        pos_embed_user = self.m_output_attr_embedding_user(pos_targets)
        # pos_embed_user = F.normalize(pos_embed_user, dim=1)

        pos_embed_item = self.m_output_attr_embedding_item(pos_targets)
        # pos_embed_item = F.normalize(pos_embed_item, dim=1)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num

        pos_logits_user = self.f_get_logits(pos_embed_user, user_output)
        pos_logits_item = self.f_get_logits(pos_embed_item, item_output)

        pos_logits = pos_logits_user+pos_logits_item
        neg_logits = neg_logits_user+neg_logits_item

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        # logits += attr_tf_user*scale_weight
        # logits += attr_tf_item*scale_weight

        # logits += attr_tf_user
        # logits += attr_tf_item
    
        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

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