import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        # print("dec")
        # Q_batch_size = Q.size(0)
        # print("Q_batch_size", Q_batch_size)

        # print("A", A[:Q_batch_size].size())
        # print(A[:Q_batch_size])
        # print("A size", A.size())

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.m_enc = nn.Sequential(
        ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln), 
        ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))

        self.m_dec = nn.Sequential(
        nn.Dropout(),
        PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        nn.Dropout(),
        nn.Linear(dim_hidden, dim_output))

    def forward(self, x):
        return self.m_dec(self.m_enc(x)).squeeze()

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights

import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights

class SET_ENCODER(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads=4):
        super(SET_ENCODER, self).__init__()

        self.m_dropout = nn.Dropout()
        
        encoder_layers = TransformerEncoderLayer(dim_input, num_heads, dim_output)
        self.m_enc = TransformerEncoder(encoder_layers, 2)

    def forward(self, x, x_mask):
        x = x.transpose(0, 1)
        x_enc, w = self.m_enc(x, src_key_padding_mask=x_mask)
        x_enc = x_enc.transpose(0, 1)

        return x_enc

class SET_DECODER(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads=4):
        super(SET_DECODER, self).__init__()

        # self.m_mab = MAB(dim_input, dim_input, dim_input, num_heads)

        self.m_dropout = nn.Dropout()
      
        self.m_output_linear = nn.Linear(dim_input, dim_output)

        self.f_init_weight()

    def f_init_weight(self):
        initrange = 0.1

        torch.nn.init.uniform_(self.m_output_linear.weight, -initrange, initrange)
        if self.m_output_linear.bias is not None:
            torch.nn.init.uniform_(self.m_output_linear.bias, -initrange, initrange)

    def forward(self, x, x_mask, context):

        ### x_mask: batch_size*seq_len
        ### context: batch_size*embed_size
        ### x: batch_size*seq_len*embed_size
        x = self.m_dropout(x)

        ### attn_weight: batch_size*seq_len*1
        attn_weight = torch.matmul(x*x_mask.unsqueeze(-1), context.unsqueeze(-1))

        ### x_output: batch_size*1
        x_output = torch.sum(attn_weight*x_mask.unsqueeze(-1), dim=1)
        
        return x_output

    def f_eval_forward(self, x, context):

        ### x_mask: batch_size*seq_len
        ### context: batch_size*embed_size
        ### x: batch_size*seq_len*embed_size 
        
        ### attn_weight: batch_size*seq_len*1
        attn_weight = torch.matmul(x, context.unsqueeze(-1))

        ### x_output: batch_size*1
        x_output = torch.sum(attn_weight, dim=1)
        
        return x_output

class SetTransformer2(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer2, self).__init__()
        encoder_layers = TransformerEncoderLayer(dim_input, num_heads, dim_output)
        self.m_enc = TransformerEncoder(encoder_layers, 2)

        self.m_dec = nn.Sequential(
        nn.Dropout(),
        PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        nn.Dropout(),
        nn.Linear(dim_hidden, dim_output))

    def forward(self, x, x_mask):
        x = x.transpose(0, 1)
        x_enc, w = self.m_enc(x, src_key_padding_mask=x_mask)
        x_enc = x_enc.transpose(0, 1)

        x_dec = self.m_dec(x_enc)
        
        x_dec = x_dec.squeeze()

        return x_dec, w

class ATTR_TRANSFORMER(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(ATTR_TRANSFORMER, self).__init__()
        
        self.m_dropout = nn.Dropout()

        self.m_attr_attn = nn.Linear(dim_input, dim_hidden)
        self.m_user_attn = nn.Linear(dim_input, dim_hidden)

        # self.m_context = nn.Linear(dim_hidden, 1, bias=False)

        self.m_softmax = nn.Softmax(dim=1)
        
    def f_init_weight(self):
        initrange = 0.1

        torch.nn.init.uniform_(self.m_attr_attn.weight, -initrange, initrange)
        if self.m_attr_attn.bias is not None:
            torch.nn.init.uniform_(self.m_attr_attn.bias, -initrange, initrange)

        torch.nn.init.uniform_(self.m_user_attn.weight, -initrange, initrange)
        if self.m_user_attn.bias is not None:
            torch.nn.init.uniform_(self.m_user_attn.bias, -initrange, initrange)

    def forward(self, x, x_mask, user, user_size):

        ### x_mask: batch_size*set_size
        ### x: batch_size*set_size*embed_size
        x = self.m_dropout(x)

        ### attn_x: batch_size*set_size*hidden_size
        attn_x = torch.tanh(self.m_attr_attn(x))
        attn_user = self.m_user_attn(user)

        # print("x_mask", x_mask.size())
        # print("attn_user", attn_user.size())
        # print("attn_x", attn_x.size())
        # print("user_size", user_size.size())
        # print("user_size", user_size)
        repeat_user = attn_user.repeat_interleave(user_size, dim=0)

        attn_weight = self.m_softmax(torch.matmul(attn_x*x_mask.unsqueeze(-1), repeat_user.unsqueeze(-1)).squeeze(-1))

        ### attn_weighted_x: batch_size*set_size*dim_hidden
        attn_weighted_x = torch.sum(attn_x*(attn_weight*x_mask).unsqueeze(-1), dim=1)
        # print("attn_weighted_x", attn_weighted_x.size())

        # acc_user_size = torch.cumsum(user_size, dim=0)
        # last_size = 0

        return attn_weighted_x, attn_weight