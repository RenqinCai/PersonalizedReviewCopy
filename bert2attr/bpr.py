import torch
from torch import log
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class BPR(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(BPR, self).__init__()

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

        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        # self.m_user_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_item_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size*2)

        self.m_output_linear = nn.Sequential(nn.Linear(self.m_attr_embed_size*2, self.m_attr_embed_size*4), nn.Sigmoid(), nn.Linear(self.m_attr_embed_size*4, self.m_attr_embed_size*2))


        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        torch.nn.init.normal_(self.m_output_attr_embedding.weight, 0.0, 0.1)
        torch.nn.init.normal_(self.m_user_embedding.weight, 0.0, 0.1)
        torch.nn.init.normal_(self.m_item_embedding.weight, 0.0, 0.1)

    # def f_init_weight(self):
    #     torch.nn.init.normal_(self.m_output_attr_embedding.weight, 0.0, 1e-5)
    #     torch.nn.init.normal_(self.m_user_embedding.weight, 0.0, 1e-5)
    #     torch.nn.init.normal_(self.m_item_embedding.weight, 0.0, 1e-5)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, item_ids, user_ids, pos_targets, pos_lens, neg_targets, neg_lens):
        # print("==="*10)

        """user"""
        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)

        """item"""
        ### item_x: batch_size*item_embed
        item_embed = self.m_item_embedding(item_ids)
        
        user_output = user_embed
        item_output = item_embed
        
        output = torch.cat([user_output, item_output], dim=1)
        user_item_output = output
        # user_item_output = self.m_output_linear(output)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        neg_embed = self.m_output_attr_embedding(neg_targets)
        neg_logits = torch.matmul(neg_embed, user_item_output.unsqueeze(-1))
        neg_logits = neg_logits.squeeze(-1)

        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### pos_targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        ### user_item_output: batch_size*hidden_size*1
        ### pos_logits: batch_size*pos_num
        pos_embed = self.m_output_attr_embedding(pos_targets)
        pos_logits = torch.matmul(pos_embed, user_item_output.unsqueeze(-1))
        pos_logits = pos_logits.squeeze(-1)

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, item_ids, user_ids):

        """ item """
        ### item_x: batch_size*item_embed
        item_embed = self.m_item_embedding(item_ids)
        
        """ user """
        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)

        user_output = user_embed
        item_output = item_embed

        output = torch.cat([user_output, item_output], dim=1)
        # user_item_output = output
        # output = self.m_output_linear(output)

        logits = torch.matmul(output, self.m_output_attr_embedding.weight.t())

        return logits