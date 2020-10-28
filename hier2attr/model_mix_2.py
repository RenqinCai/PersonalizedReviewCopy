"""
use  [user, user_x, item, item_x]
use user, item attr indicator
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

        self.m_ind_embedding = nn.Embedding(3, self.m_attr_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*2)
        self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*2)
        
        self.m_feat_num = 13
        self.m_feat_attrweight = nn.Sequential(nn.Linear(self.m_feat_num, 1), nn.Sigmoid())

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

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
        
        attr_ind_embed = self.m_ind_embedding(attr_inds)

        attr_mask = self.f_generate_mask(attr_lens)

        ### attr_attn: batch_size*attr_lens*embed_size
        attr_attn = attr_embed+attr_ind_embed
        attr_attn = attr_attn.transpose(0, 1)
        attr_attn = self.m_attn(attr_attn, src_key_padding_mask=attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        ### attr_feat: batch_size*feat_num
        ### attr_weight: batch_size*1
        attr_weight = self.m_feat_attrweight(attr_feat)

        ### attr_tf: batch_size*attr_lens
        attr_user_weight = attr_weight.squeeze(-1)        
        attr_item_weight = 1.0 - attr_user_weight

        attr_tf = attr_tf.unsqueeze(-1)
        # attr_user_mask = self.f_generate_mask(attr_lens_user)
        # attr_item_mask = self.f_generate_mask(attr_lens_item)

        attr_mask = self.f_generate_mask(attr_lens)

        user_x = []
        item_x = []

        batch_size = attr_attn.size(0)
        for i in range(batch_size):

            attr_lens_user_i = attr_lens_user[i]
            attr_lens_item_i = attr_lens_item[i]
            
            attr_user_weight_i = attr_user_weight[i]
            tmp_user = attr_attn[i][:attr_lens_user_i]*attr_user_weight_i*attr_tf[i][:attr_lens_user_i]

            attr_item_weight_i = attr_item_weight[i]
            tmp_item = attr_attn[i][attr_lens_user_i:attr_lens_user_i+attr_lens_item_i]*attr_item_weight_i*attr_tf[i][attr_lens_user_i:attr_lens_user_i+attr_lens_item_i]

            avg_attr_user_i = tmp_user.sum(0)/attr_lens_user_i
            avg_attr_item_i = tmp_item.sum(0)/attr_lens_item_i

            user_x.append(avg_attr_user_i.unsqueeze(0))
            item_x.append(avg_attr_item_i.unsqueeze(0))

        user_x = torch.cat(user_x, dim=0)
        item_x = torch.cat(item_x, dim=0)

        # user_output = user_x
        # item_output = item_x

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        user_output = torch.cat([user_x, user_embed], dim=-1)
        item_output = torch.cat([item_x, item_embed], dim=-1)

        neg_embed_user = self.m_output_attr_embedding_user(neg_targets)
        neg_embed_item = self.m_output_attr_embedding_item(neg_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        neg_logits_user = torch.matmul(neg_embed_user, user_output.unsqueeze(-1))
        neg_logits_user = neg_logits_user.squeeze(-1)

        neg_logits_item = torch.matmul(neg_embed_item, item_output.unsqueeze(-1))
        neg_logits_item = neg_logits_item.squeeze(-1)

        # print("neg_lens", neg_lens)
        # exit()
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        # pos_embed = self.m_attr_embedding(pos_targets)

        pos_embed_user = self.m_output_attr_embedding_user(pos_targets)
        pos_embed_item = self.m_output_attr_embedding_item(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits_user = torch.matmul(pos_embed_user, user_output.unsqueeze(-1))
        pos_logits_user = pos_logits_user.squeeze(-1)

        pos_logits_item = torch.matmul(pos_embed_item, item_output.unsqueeze(-1))
        pos_logits_item = pos_logits_item.squeeze(-1)

        pos_logits = pos_logits_user+pos_logits_item
        neg_logits = neg_logits_user+neg_logits_item

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, attr, attr_inds, attr_tf, attr_feat, attr_lens, attr_lens_user, attr_lens_item, user_ids, item_ids):

        attr_embed = self.m_attr_embedding(attr)
        
        attr_ind_embed = self.m_ind_embedding(attr_inds)

        attr_mask = self.f_generate_mask(attr_lens)

        ### attr_attn: batch_size*attr_lens*embed_size
        attr_attn = attr_embed+attr_ind_embed
        attr_attn = attr_attn.transpose(0, 1)
        attr_attn = self.m_attn(attr_attn, src_key_padding_mask=attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        ### attr_tf: batch_size*attr_lens
        ### attr_feat: batch_size*feat_num
        ### attr_weight: batch_size*1
        attr_weight = self.m_feat_attrweight(attr_feat)
        
        attr_user_weight = attr_weight.squeeze(-1)
        attr_item_weight = 1.0 - attr_user_weight

        attr_tf = attr_tf.unsqueeze(-1)

        # attr_user_mask = self.f_generate_mask(attr_lens_user)
        # attr_item_mask = self.f_generate_mask(attr_lens_item)

        attr_mask = self.f_generate_mask(attr_lens)

        user_x = []
        item_x = []

        batch_size = attr_attn.size(0)
        for i in range(batch_size):

            attr_lens_user_i = attr_lens_user[i]
            attr_lens_item_i = attr_lens_item[i]
            
            attr_user_weight_i = attr_user_weight[i]
            tmp_user = attr_attn[i][:attr_lens_user_i]*attr_user_weight_i*attr_tf[i][:attr_lens_user_i]

            attr_item_weight_i = attr_item_weight[i]
            tmp_item = attr_attn[i][attr_lens_user_i:attr_lens_user_i+attr_lens_item_i]*attr_item_weight_i*attr_tf[i][attr_lens_user_i:attr_lens_user_i+attr_lens_item_i]

            avg_attr_user_i = tmp_user.sum(0)/attr_lens_user_i
            avg_attr_item_i = tmp_item.sum(0)/attr_lens_item_i

            user_x.append(avg_attr_user_i.unsqueeze(0))
            item_x.append(avg_attr_item_i.unsqueeze(0))

        user_x = torch.cat(user_x, dim=0)
        item_x = torch.cat(item_x, dim=0)

        # user_output = user_x
        # item_output = item_x

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        user_output = torch.cat([user_x, user_embed], dim=-1)
        item_output = torch.cat([item_x, item_embed], dim=-1)

        # user_output = (1.0-self.m_gamma)*user_embed+self.m_gamma*item_attr_user_output
        # item_output = (1.0-self.m_gamma)*item_embed+self.m_gamma*user_attr_item_output

        logits_user = torch.matmul(user_output, self.m_output_attr_embedding_user.weight.t())
        logits_item = torch.matmul(item_output, self.m_output_attr_embedding_item.weight.t())

        logits = logits_user+logits_item

        return logits