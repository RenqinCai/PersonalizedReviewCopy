import enum
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

        # self.m_mix_tanh = nn.Tanh()
        # self.m_mix_af = nn.ReLU()
        # self.m_output = nn.Linear(self.m_output_hidden_size, 1)
        # self.m_output = nn.Linear(1, 1)

        self.m_lambda = 0.28
        self.m_user_output = nn.Linear(1, 1)
        self.m_attr_item_linear = nn.Linear(1, 1)
        self.m_attr_user_linear = nn.Linear(1, 1)
    
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

    def forward(self, attr_item, attr_tf_item, attr_lens_item, attr_index_item, item_ids, attr_user, attr_tf_user, attr_lens_user, attr_index_user, user_ids, lens, logits):

        mask = self.f_generate_mask(lens)

        attr_item_x = self.m_attr_embedding(attr_item)
        attr_item_x = attr_item_x.transpose(0, 1)
        # print("attr_item_x", attr_item_x.size())
        # print(attr_item_x)

        attr_item_mask = self.f_generate_mask(attr_lens_item)
        # print("attr_item_mask", attr_item_mask.size())
        # print(attr_item_mask)

        attr_attn_item = self.m_attn(attr_item_x, src_key_padding_mask = attr_item_mask)
        attr_attn_item = attr_attn_item.transpose(0, 1)

        attr_item = self.m_attr_linear(attr_attn_item)

        user_x = self.m_user_embedding(user_ids)
        user_x = self.m_user_linear(user_x)

        user_x = user_x.unsqueeze(2)
        
        user_attr_item = torch.matmul(attr_item, user_x).squeeze()
        # print("user_attr_item", user_attr_item.size())
        # user_attr_item_logits = user_attr_item.unsqueeze(2)

        attr_item_weight = attr_tf_item.unsqueeze(2)
        attr_item_logits = self.m_attr_item_linear(attr_item_weight)

        # user_attr_item_logits = user_attr_item_logits.squeeze(-1)
        attr_item_logits = attr_item_logits.squeeze(-1)
        user_attr_item_logits = user_attr_item + attr_item_logits

        # """ user """

        # attr_user_x = self.m_attr_embedding(attr_user) 
        # attr_user_x = attr_user_x.transpose(0, 1)

        # attr_user_mask = self.f_generate_mask(attr_lens_user)
    
        # attr_attn_user = self.m_attn(attr_user_x, src_key_padding_mask=attr_user_mask)
        # attr_attn_user = attr_attn_user.transpose(0, 1)

        # attr_user = self.m_attr_linear(attr_attn_user)

        # item_x = self.m_item_embedding(item_ids)
        # item_x = self.m_item_linear(item_x)
        # item_x = item_x.unsqueeze(2)

        # item_attr_user = torch.matmul(attr_user, item_x).squeeze()

        # # item_attr_user_logits = item_attr_user.unsqueeze(2)

        # attr_user_weight = attr_tf_user.unsqueeze(2)
        # attr_user_logits = self.m_attr_user_linear(attr_user_weight)

        # # item_attr_user_logits = item_attr_user_logits.squeeze(-1)
        # attr_user_logits = attr_user_logits.squeeze(-1)
        # item_attr_user_logits = item_attr_user + attr_user_logits

        batch_size = attr_item.size(0)
        len_i = attr_index_item.size(1)

        tmp_logits = torch.zeros_like(logits)
        tmp_logits.requires_grad = False

        # print("=="*15)
        # print(logits.requires_grad)
        # print(tmp_logits.requires_grad)

        # print(user_attr_item_logits.size())
        # print(user_attr_item_logits)
        # print(len_i)
        # print(attr_index_item.size())
        # print(attr_index_item)
        for i in range(len_i):
            # print("--"*10, i)
        # for i, val in enumerate(attr_index_item):
            val = attr_index_item[:, i]

            tmp_logits[torch.arange(batch_size), val] = tmp_logits[torch.arange(batch_size), val] + user_attr_item_logits[:, i]*(~attr_item_mask[:, i])

            # print("tmp_logits 2")
            # print(tmp_logits)
        # len_j = attr_index_user.size(1)
        # for j in range(len_j):
            
        #     val = attr_index_user[:, j]
        #     tmp_logits[torch.arange(batch_size), val] = tmp_logits[torch.arange(batch_size), val] + item_attr_user_logits[:, j]*attr_user_mask[:, j]

        # print("xx"*15)
        # print(logits.requires_grad)
        # print(tmp_logits.requires_grad)

        logits = tmp_logits
        # logits.data = tmp_logits.data
        # print("++"*15)
        # print(logits.requires_grad)
        # print(tmp_logits.requires_grad)
        # print(logits)
        # print(tmp_logits)
        # # logits = logits+1
        # logits.scatter_(1, attr_index_item, user_attr_item_logits)
        # logits.scatter_(1, attr_index_user, item_attr_user_logits)
        # exit()
        # if torch.all(torch.eq(user_attr_item_logits, logits)):
            # print("not equal")
        # debug = torch.eq(user_attr_item_logits, logits)
        # print(debug)
        # print(~attr_item_mask)

        # exit()
        return user_attr_item_logits, attr_item_mask, None, None, logits, mask
        # return user_attr_item_logits, attr_item_mask, item_attr_user_logits, attr_user_mask, logits, mask