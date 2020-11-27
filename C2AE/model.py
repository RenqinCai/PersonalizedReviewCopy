import numpy as np
import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class _ATTR_NETWORK(nn.Module):
    def __init__(self, vocab, args, device):
        super(_ATTR_NETWORK, self).__init__()

        self.m_device = device
        
        self.m_vocab_size = vocab.vocab_size
        self.m_user_num = vocab.user_num
        self.m_item_num = vocab.item_num

        print("vocab size", self.m_vocab_size)

        self.m_attr_embed_size = args.attr_emb_size
        self.m_hidden_size = args.output_hidden_size

        # self.m_attn_head_num = args.attn_head_num
        # self.m_attn_layer_num = args.attn_layer_num
        # self.m_attn_linear_size = args.attn_linear_size

        self.m_feature_dim = 500

        negative_slope = 0.1

        self.m_hidden_1 = self.m_hidden_size
        self.m_hidden_2 = self.m_hidden_size

        # self.m_user_embed = torch.nn.Parameter(torch.zeros(self.m_vocab_size, self.m_feature_dim))
        self.m_user_embed = nn.Embedding(self.m_user_num, self.m_feature_dim)
        init_weight_linear_relu(self.m_user_embed.weight)

        # self.m_item_embed = torch.nn.Parameter(torch.zeros(self.m_vocab_size, self.m_feature_dim))
        self.m_item_embed = nn.Embedding(self.m_item_num, self.m_feature_dim)
        init_weight_linear_relu(self.m_item_embed.weight)

        self.m_embed_ac = nn.LeakyReLU(negative_slope=negative_slope)

        """Fx"""
        # self.m_Fx_w0 = torch.nn.Parameter(torch.zeros(self.m_feature_dim*2, self.m_hidden_1))
        # init_weight_linear_relu(self.m_Fx_w0)
        # self.m_Fx_b0 = torch.nn.Parameter(torch.zeros(self.m_hidden_1))
        # init_bias_linear_relu(self.m_Fx_b0)
        # self.m_Fx_ac_0 = nn.LeakyReLU(negative_slope=negative_slope)

        # self.m_Fx_w1 = torch.nn.Parameter(torch.zeros(self.m_hidden_1, self.m_hidden_1))
        # init_weight_linear_relu(self.m_Fx_w1)
        # self.m_Fx_b1 = torch.nn.Parameter(torch.zeros(self.m_hidden_1))
        # init_bias_linear_relu(self.m_Fx_b1)
        # self.m_Fx_ac_1 = nn.LeakyReLU(negative_slope=negative_slope)

        # self.m_Fx_w2 = torch.nn.Parameter(torch.zeros(self.m_hidden_1, self.m_attr_embed_size))
        self.m_Fx_w2 = torch.nn.Parameter(torch.zeros(self.m_feature_dim*2, self.m_attr_embed_size))
        init_weight_linear_sigmoid(self.m_Fx_w2)
        self.m_Fx_b2 = torch.nn.Parameter(torch.zeros(self.m_attr_embed_size))
        init_bias_linear_sigmoid(self.m_Fx_b2)
        self.m_Fx_ac_2 = nn.Sigmoid()

        # self.m_Fx_layer_norm = torch.nn.LayerNorm(self.m_attr_embed_size)
        

        """Fe"""
        self.m_Fe_w0 = torch.nn.Parameter(torch.zeros(self.m_vocab_size, self.m_hidden_1))
        init_weight_linear_relu(self.m_Fe_w0)
        self.m_Fe_b0 = torch.nn.Parameter(torch.zeros(self.m_hidden_1))
        init_bias_linear_relu(self.m_Fe_b0)
        self.m_Fe_ac_0 = nn.LeakyReLU(negative_slope=negative_slope)

        self.m_Fe_w1 = torch.nn.Parameter(torch.zeros(self.m_hidden_1, self.m_attr_embed_size))
        init_weight_linear_sigmoid(self.m_Fe_w1)
        self.m_Fe_b1 = torch.nn.Parameter(torch.zeros(self.m_attr_embed_size))
        init_bias_linear_sigmoid(self.m_Fe_b1)
        self.m_Fe_ac_1 = nn.Sigmoid()

        # self.m_Fe_layer_norm = torch.nn.LayerNorm(self.m_attr_embed_size)

        """Fd"""
        self.m_Fd_w0 = torch.nn.Parameter(torch.zeros(self.m_attr_embed_size, self.m_hidden_1))
        init_weight_linear_relu(self.m_Fd_w0)
        self.m_Fd_b0 = torch.nn.Parameter(torch.zeros(self.m_hidden_1))
        init_bias_linear_relu(self.m_Fd_b0)
        self.m_Fd_ac_0 = nn.LeakyReLU(negative_slope=negative_slope)

        self.m_Fd_w1 = torch.nn.Parameter(torch.zeros(self.m_hidden_1, self.m_vocab_size))
        init_weight_linear_sigmoid(self.m_Fd_w1)
        self.m_Fd_b1 = torch.nn.Parameter(torch.zeros(self.m_vocab_size))
        init_bias_linear_sigmoid(self.m_Fd_b1)
        self.m_Fd_ac_1 = nn.Sigmoid()

        self = self.to(self.m_device)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_encode_y(self, input, batch_index, epoch_index):
        ### pos_target_embed: batch_size*attr_embed

        y_e = torch.matmul(input, self.m_Fe_w0)+self.m_Fe_b0

        y_e = self.m_Fe_ac_0(y_e)

        Fe = torch.matmul(y_e, self.m_Fe_w1)+self.m_Fe_b1

        Fe = self.m_Fe_ac_1(Fe)

        # Fe = self.m_Fe_layer_norm(Fe)

        # if epoch_index < 2:
        #     if batch_index < 2:
        #         print("..."*2+"Fe"+"..."*2, Fe)

        Fe = Fe-0.4

        # if epoch_index < 2:
        #     if batch_index < 2:
        #         print("Fe", Fe)

        return Fe
        
    def f_encode_x(self, user, item, batch_index, epoch_index):
        # user = user.float()
        # print("user", user.size())
        # print(self.m_user_embed.size())
        # user_embed = torch.matmul(user, self.m_user_embed)
        user_embed = self.m_user_embed(user)

        # item = item.float()
        # print("item", item.size())
        # print(self.m_item_embed.size())
        # item_embed = torch.matmul(item, self.m_item_embed)
        item_embed = self.m_item_embed(item)

        x = torch.cat([user_embed, item_embed], dim=-1)

        x = self.m_embed_ac(x)

        # x = torch.matmul(x, self.m_Fx_w0)+self.m_Fx_b0

        # x = self.m_Fx_ac_0(x)

        # x = torch.matmul(x, self.m_Fx_w1)+self.m_Fx_b1

        # x = self.m_Fx_ac_1(x)

        x = torch.matmul(x, self.m_Fx_w2)+self.m_Fx_b2

        x = self.m_Fx_ac_2(x)

        # x = self.m_Fx_layer_norm(x)

        # if epoch_index < 2:
        #     if batch_index < 2:
        #         print("..."*2+"x"+"..."*2, x)

        x = x-0.4
        # if epoch_index < 2:
        #     if batch_index < 2:
        #         print("x", x)

        return x

    def f_decode_y(self, input):
        # print("input", input.size())

        Fd = torch.matmul(input, self.m_Fd_w0)+self.m_Fd_b0

        Fd = self.m_Fd_ac_0(Fd)

        Fd = torch.matmul(Fd, self.m_Fd_w1)+self.m_Fd_b1

        Fd = self.m_Fd_ac_1(Fd)

        # Fd = self.m_Fd_linear_0(input)
        # Fd = self.m_Fd_ac_0(Fd)

        # Fd = self.m_Fd_linear_1(Fd)
        # Fd = self.m_Fd_ac_1(Fd)

        return Fd

    def forward(self, user, item, y, batch_index, epoch_index):
        
        # print("x train", x.size())
        # print("y train", y.size())
        # print("y", y)
        enc_y = self.f_encode_y(y, batch_index, epoch_index)
        # print("enc y", enc_y.size())

        logits = self.f_decode_y(enc_y)

        enc_x = self.f_encode_x(user, item, batch_index, epoch_index)
        
        return enc_x, enc_y, logits

    def f_eval_forward(self, user, item, epoch_index):
        # print("x eval", x.size())
        enc_x = self.f_encode_x(user, item, 100, epoch_index)

        # print("enc x", enc_x.size())

        logits = self.f_decode_y(enc_x)
        
        return logits

def init_weight_linear_relu(m):
    with torch.no_grad():
        weight_size = m.size()
        randw = np.random.rand(weight_size[0], weight_size[1])
        # print("... randw 1", randw)

        randw = randw.flatten(order="C").reshape((weight_size[0], weight_size[1]), order="F")
        randw = randw.astype(np.float32)
        randw = 2*(randw-0.5)*0.01

        # print("rand w relu", randw)

        m.data.copy_(torch.from_numpy(randw))

def init_bias_linear_relu(m):
    with torch.no_grad():
        bias_size = m.size()
        randb = np.random.rand(bias_size[0])
        randb = randb.astype(np.float32)
        randb = randb*0.1

        m.data.copy_(torch.from_numpy(randb))

def init_weight_linear_sigmoid(m):
    with torch.no_grad():
        weight_size = m.size()
        randw = np.random.rand(weight_size[0], weight_size[1]).flatten(order="C").reshape((weight_size[0], weight_size[1]), order="F")
        randw = randw.astype(np.float32)
        randw = 8*(randw-0.5)*np.sqrt(6)/np.sqrt(weight_size[0]+weight_size[1])

        # print("rand w sigmoid", randw)

        m.data.copy_(torch.from_numpy(randw))

def init_bias_linear_sigmoid(m):
    with torch.no_grad():
        nn.init.zeros_(m)
