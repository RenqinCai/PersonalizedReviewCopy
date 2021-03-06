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
        self.m_item_output = nn.Linear(1, 1)
    
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

    def forward(self, attr_input, attr_tf_input, lens, user_ids, item_ids):
        # print("=="*10)
        # print("attr_tf_input", attr_tf_input)
        # exit()

        attr_x = self.m_attr_embedding(attr_input)
        attr_x = attr_x.transpose(0, 1)
        
        mask = self.f_generate_mask(lens)

        attr_attn_x = self.m_attn(attr_x, src_key_padding_mask = mask)
        attr_attn_x = attr_attn_x.transpose(0, 1)

        attr_x = self.m_attr_linear(attr_attn_x)

        user_x = self.m_user_embedding(user_ids)
        user_x = self.m_user_linear(user_x)

        user_x = user_x.unsqueeze(2)

        user_attr_weight = torch.matmul(attr_x, user_x).squeeze()
    
        # user_attr_weight = F.sigmoid(torch.matmul(attr_x, user_x).squeeze(), dim=-1)
        # user_attr_weight = F.softmax(user_attr_weight, dim=-1)

        user_attr_weight = user_attr_weight.unsqueeze(2)
        ### user_attr_logits: batch_size*seq_len*1
        user_attr_logits = self.m_user_output(user_attr_weight)

        # user_item_attr_logits = user_attr_logits
        ### attr_tf_input: batch_size*seq_len
        ### item_attr_logits: batch_size*seq_len
        # item_attr_weight = F.softmax(attr_tf_input, dim=-1)
        item_attr_weight = attr_tf_input.unsqueeze(2)
        item_attr_logits = self.m_item_output(item_attr_weight)

        user_attr_logits = user_attr_logits.squeeze(-1)
        item_attr_logits = item_attr_logits.squeeze(-1)
        # print("user_attr_logits", user_attr_logits.size())
        # print("item_attr_logits", item_attr_logits.size())

        # print("lens", lens)
        # print("user_attr_logits", user_attr_logits)
        # print("item_attr_logits", item_attr_logits)
        # exit()
        user_item_attr_logits = user_attr_logits + item_attr_logits
        # user_item_attr_logits = user_attr_logits*item_attr_logits
        # user_item_attr_logits = self.m_lambda*user_attr_logits+(1-self.m_lambda)*item_attr_logits

        return user_item_attr_logits, mask