import enum
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

        self.m_attr_user_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_attr_item_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_output_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.01
        # torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        # torch.nn.init.uniform_(self.m_output_attr_embedding_user_x.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_item_x.weight, -initrange, initrange)
        
        torch.nn.init.uniform_(self.m_attr_embedding_x.weight, -initrange, initrange)
        # torch.nn.init.uniform_(self.m_output_attr_embedding_x.weight, -initrange, initrange)

        # torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_attr_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_attr_item_embedding.weight, -initrange, initrange)
        
        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, user_ids, item_ids, attr_ids, attr_len):

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        attr_embed = self.m_attr_embedding_x(attr_ids)

        attr_len = attr_len+2
        attr_mask = self.f_generate_mask(attr_len)

        input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1), attr_embed], dim=1)

        input_embed = input_embed.transpose(0, 1)
        attr_attn = self.m_attn(input_embed, src_key_padding_mask = attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        attr_attn = attr_attn*(~attr_mask.unsqueeze(-1))

        user_hidden = attr_attn[:, 0]
        item_hidden = attr_attn[:, 1]

        # user_hidden = user_embed
        # item_hidden = item_embed

        ### voc_size*embed_size
        voc_user_embed = self.m_attr_user_embedding.weight
        voc_item_embed = self.m_attr_item_embedding.weight

        user_logits = torch.matmul(user_hidden, voc_user_embed.t())

        item_logits = torch.matmul(item_hidden, voc_item_embed.t())

        logits = user_logits+item_logits

        return logits

    def f_eval(self, user_ids, item_ids, attr_ids, attr_len, top_k):

        user_embed = self.m_user_embedding(user_ids)
        item_embed = self.m_item_embedding(item_ids)

        # if torch.sum(attr_len) == 0:
            
        #     input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)

        #     input_embed = input_embed.transpose(0, 1)
        #     attr_attn = self.m_attn(input_embed)
        #     attr_attn = attr_attn.transpose(0, 1)
        # else:

        attr_embed = self.m_attr_embedding_x(attr_ids)

        attr_len = attr_len+2
        attr_mask = self.f_generate_mask(attr_len)

        input_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1), attr_embed], dim=1)

        input_embed = input_embed.transpose(0, 1)
        attr_attn = self.m_attn(input_embed, src_key_padding_mask = attr_mask)
        attr_attn = attr_attn.transpose(0, 1)

        attr_attn = attr_attn*(~attr_mask.unsqueeze(-1))

        user_hidden = attr_attn[:, 0]
        item_hidden = attr_attn[:, 1]

        # user_hidden = user_embed
        # item_hidden = item_embed

        ### voc_size*embed_size
        voc_user_embed = self.m_attr_user_embedding.weight
        voc_item_embed = self.m_attr_item_embedding.weight

        user_logits = torch.matmul(user_hidden, voc_user_embed.t())

        item_logits = torch.matmul(item_hidden, voc_item_embed.t())

        logits = user_logits+item_logits

        logits = logits.view(-1, logits.size(1))
        _, preds = torch.topk(logits, top_k, dim=-1)

        return preds

