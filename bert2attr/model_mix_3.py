"""
[user, attr_x, item] without softmax
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

        self.m_attn_linear_size = args.attn_linear_size

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        self.m_output_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*3)
        
        self.m_feat_num = 13
        self.m_feat_attrweight = nn.Sequential(nn.Linear(self.m_feat_num, 1), nn.Sigmoid())

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
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

    def forward(self, attr, attr_inds, attr_tf, attr_feat, attr_lens, attr_lens_user, attr_lens_item, user_ids, item_ids, pos_targets, pos_lens, neg_targets, neg_lens):
        
        attr_embed = self.m_attr_embedding(attr)
        
        attr_mask = self.f_generate_mask(attr_lens)

        ### attr_attn: batch_size*attr_lens*embed_size
        attr_attn = attr_embed.transpose(0, 1)
        attr_attn = self.m_attn(attr_attn, src_key_padding_mask=attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        ### attr_tf: batch_size*attr_lens
        ### attr_feat: batch_size*feat_num
        ### attr_weight: batch_size*1
        attr_weight = self.m_feat_attrweight(attr_feat)
        
        attr_user_weight = attr_weight.squeeze(-1)
        attr_item_weight = 1.0 - attr_user_weight

        # attr_tf = attr_tf

        weight = []
        batch_size = attr_attn.size(0)
        for i in range(batch_size):

            attr_lens_user_i = attr_lens_user[i]
            attr_lens_item_i = attr_lens_item[i]
            
            attr_user_weight_i = attr_user_weight[i]
            attr_user_tf_weight_i = attr_user_weight_i*attr_tf[i][:attr_lens_user_i]

            attr_item_weight_i = attr_item_weight[i]
            attr_item_tf_weight_i = attr_item_weight_i*attr_tf[i][attr_lens_user_i:]

            attr_weight_i = torch.cat([attr_user_tf_weight_i, attr_item_tf_weight_i], dim=-1)

            weight.append(attr_weight_i.unsqueeze(0))

        weight = torch.cat(weight, dim=0)
        # normalized_weight = F.softmax(weight, dim=-1)

        # normalized_weight = normalized_weight.unsqueeze(-1)
        attr_mask = (weight*~attr_mask).unsqueeze(-1)

        masked_attr_embed = attr_attn*attr_mask
        attr_x = masked_attr_embed.sum(1)

        # attr_x = masked_attr_embed.sum(1)/((~attr_mask).sum(1).unsqueeze(-1))

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)
        output = torch.cat([user_embed, attr_x, item_embed], dim=-1)
      
        neg_embed = self.m_output_attr_embedding(neg_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        neg_logits = torch.matmul(neg_embed, output.unsqueeze(-1))
        neg_logits = neg_logits.squeeze(-1)

        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        # pos_embed = self.m_attr_embedding(pos_targets)

        pos_embed = self.m_output_attr_embedding(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits = torch.matmul(pos_embed, output.unsqueeze(-1))
        pos_logits = pos_logits.squeeze(-1)

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, attr, attr_inds, attr_tf, attr_feat, attr_lens, attr_lens_user, attr_lens_item, user_ids, item_ids):

        attr_embed = self.m_attr_embedding(attr)
        
        attr_mask = self.f_generate_mask(attr_lens)

        ### attr_attn: batch_size*attr_lens*embed_size
        attr_attn = attr_embed.transpose(0, 1)
        attr_attn = self.m_attn(attr_attn, src_key_padding_mask=attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        ### attr_tf: batch_size*attr_lens
        ### attr_feat: batch_size*feat_num
        ### attr_weight: batch_size*1
        attr_weight = self.m_feat_attrweight(attr_feat)
        
        attr_user_weight = attr_weight.squeeze(-1)
        attr_item_weight = 1.0 - attr_user_weight

        # attr_tf = attr_tf.unsqueeze(-1)

        weight = []
        batch_size = attr_attn.size(0)
        for i in range(batch_size):

            attr_lens_user_i = attr_lens_user[i]
            attr_lens_item_i = attr_lens_item[i]
            
            attr_user_weight_i = attr_user_weight[i]
            attr_user_tf_weight_i = attr_user_weight_i*attr_tf[i][:attr_lens_user_i]

            attr_item_weight_i = attr_item_weight[i]
            attr_item_tf_weight_i = attr_item_weight_i*attr_tf[i][attr_lens_user_i:]

            attr_weight_i = torch.cat([attr_user_tf_weight_i, attr_item_tf_weight_i], dim=-1)

            weight.append(attr_weight_i.unsqueeze(0))

        weight = torch.cat(weight, dim=0)
        # normalized_weight = F.softmax(weight, dim=-1)
        
        # print("weight size", weight.size())
        # print("attr_mask size", attr_mask.size())
        # normalized_weight = normalized_weight.unsqueeze(-1)
        attr_mask = (weight*~attr_mask).unsqueeze(-1)

        # print("attr_mask size", attr_mask.size())

        masked_attr_embed = attr_attn*attr_mask

        attr_x = masked_attr_embed.sum(1)

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        output = torch.cat([user_embed, attr_x, item_embed], dim=-1)

        logits = torch.matmul(output, self.m_output_attr_embedding.weight.t())

        return logits