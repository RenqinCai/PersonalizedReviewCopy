import torch
from torch import log
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

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_user_linear = nn.Linear(self.m_user_embed_size, self.m_output_hidden_size)
        self.m_item_linear = nn.Linear(self.m_item_embed_size, self.m_output_hidden_size)
        self.m_attr_linear = nn.Linear(self.m_attr_embed_size, self.m_output_hidden_size)

        self.m_mix_tanh = nn.Tanh()

        self.m_output = nn.Linear(self.m_output_hidden_size, 1)
    
        self = self.to(self.m_device)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        # mask = mask.masked_fill(mask==0, float('-inf'))
        # mask = mask.masked_fill(mask==1,float(0.0))
        # mask = mask.transpose(0, 1)

        return mask

    def forward(self, attr_input, lens, user_ids, item_ids):
        attr_x = self.m_attr_embedding(attr_input)

        # attr_x = attr_x.transpose(0, 1)
        # user_x = self.m_user_embedding(user_ids)
        # item_x = self.m_item_embedding(item_ids)

        mask = self.f_generate_mask(lens)
        # print("mask size", mask.size())
        # print("attr x", attr_x.size())
        # print("mask", mask)

        # attr_attn_x = self.m_attn(attr_x, src_key_padding_mask = mask)
        # attr_attn_x = attr_attn_x.transpose(0, 1)

        # attr_x = self.m_attr_linear(attr_attn_x)
        attr_x = self.m_attr_linear(attr_x)

        # user_x = self.m_user_linear(user_x)
        # item_x = self.m_item_linear(item_x)

        # user_x = user_x.unsqueeze(1)
        # item_x = item_x.unsqueeze(1)

        # print("attr x", attr_x.size())
        # print("user x", user_x.size())
        # print("item x", item_x.size())

        # user_item_attr_x = self.m_mix_tanh(attr_x+user_x+item_x)

        user_item_attr_x = self.m_mix_tanh(attr_x)
        user_item_attr_logits = self.m_output(user_item_attr_x)

        # print("user_item_attr_logits", user_item_attr_logits.size())

        return user_item_attr_logits, mask