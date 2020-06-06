import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

class _NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device=device

        self.m_embedding_size = args.embedding_size
        self.m_user_embedding_size = args.latent_size

        self.m_hidden_size = args.hidden_size
        self.m_word_dropout_rate = args.word_dropout   
        self.m_embedding_dropout_rate = args.embedding_dropout
        self.m_latent_size = args.latent_size

        self.m_max_sequence_len = args.max_seq_length
        self.m_num_layers = args.num_layers
        self.m_bidirectional = args.bidirectional
        # self.m_rnn_type = args.rnn_type

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_unk_idx = vocab_obj.unk_idx

        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout_rate)

        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_size, self.m_latent_size)
        
        print("user size", self.m_user_size)
        print("item size", self.m_item_size)
        
        # self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=True)
        # self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size, self.m_latent_size)

        self.m_hidden2mean_s = nn.Linear(self.m_hidden_size, self.m_latent_size)
        self.m_hidden2logv_s = nn.Linear(self.m_hidden_size, self.m_latent_size)

        self.m_hidden2mean_l = nn.Linear(self.m_hidden_size, self.m_latent_size)
        self.m_hidden2logv_l = nn.Linear(self.m_hidden_size, self.m_latent_size)

        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_hidden_size)
        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

        self.m_pos_encoder = PositionalEncoding(self.m_embedding_size, self.m_embedding_dropout_rate)

        """
        encoder
        """
        self.m_en_head_num = 1
        self.m_en_layers_num = 2

        encoder_layers = TransformerEncoderLayer(self.m_embedding_size, self.m_en_head_num, self.m_hidden_size, self.m_embedding_dropout_rate)

        self.m_transformer_encoder = TransformerEncoder(encoder_layers, self.m_en_layers_num)
        
        """
        decoder
        """
        self.m_de_head_num = 1
        self.m_de_layers_num = 2

        decoder_layers = TransformerDecoderLayer(self.m_embedding_size, self.m_de_head_num, self.m_hidden_size, self.m_embedding_dropout_rate)

        self.m_transformer_decoder = TransformerDecoder(decoder_layers, self.m_de_layers_num)

        self.m_decode_strategy = args.decode

        self.m_de_mask = None

        self = self.to(self.m_device)

    def f_generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

        return mask

    def forward(self, input_sequence, input_length, input_de_sequence, input_de_length, user_ids, item_ids, random_flag):
        batch_size = input_sequence.size(0)

        input_embedding = self.m_embedding(input_sequence)
        
        input_embedding_attn = input_embedding.transpose(0, 1)

        input_embedding_attn = input_embedding_attn*math.sqrt(self.m_embedding_size)

        # input_embedding = self.m_embedding_dropout(input_embedding)

        input_embedding_attn = self.m_pos_encoder(input_embedding_attn)

        # print("input_embedding_attn shape", input_embedding_attn.size())

        encoder_outputs = self.m_transformer_encoder(input_embedding_attn, mask=None)
        
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # print("encoder outputs shape", encoder_outputs.size())

        # encoder_outputs, _ = self.m_encoder_rnn(input_embedding)

        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (input_length-1).long()
        
        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        variational_hidden = None
        z_mean = None
        z_logv = None
        z = None

        s_mean = None
        s_logv = None
        s = None

        l_mean = None
        l_logv = None
        l = None

        z_prior = None
        s_prior = None

        if random_flag == 0:
            z_mean = self.m_hidden2mean_z(last_en_hidden)
            z_logv = self.m_hidden2logv_z(last_en_hidden)
            z_std = torch.exp(0.5*z_logv)
            z = torch.randn_like(z_std)*z_std + z_mean

            z_prior = self.m_user_embedding(user_ids)

            s_mean = self.m_hidden2mean_s(last_en_hidden)
            s_logv = self.m_hidden2logv_s(last_en_hidden)
            s_std = torch.exp(0.5*s_logv)
            s = torch.randn_like(s_std)*s_std + s_mean

            s_prior = self.m_item_embedding(item_ids)

            l_mean = self.m_hidden2mean_l(last_en_hidden)
            l_logv = self.m_hidden2logv_l(last_en_hidden)
            l_std = torch.exp(0.5*l_logv)
            l = torch.randn_like(l_std)*l_std + l_mean

            # variational_hidden = z+s
            variational_hidden = z+s+l
            # variational_hidden = torch.cat([z, s, l], dim=1)
        elif random_flag== 1: 
            z_mean = self.m_hidden2mean_z(last_en_hidden)
            z_logv = self.m_hidden2logv_z(last_en_hidden)
            z_std = torch.exp(0.5*z_logv)
            z = torch.randn_like(z_std)*z_std + z_mean

            z_prior = self.m_user_embedding(user_ids)

            s_mean = self.m_hidden2mean_s(last_en_hidden)
            s_logv = self.m_hidden2logv_s(last_en_hidden)
            s_std = torch.exp(0.5*s_logv)
            s = torch.randn_like(s_std)*s_std + s_mean

            s_prior = self.m_item_embedding(item_ids)

            variational_hidden = z+s
            # variational_hidden = torch.cat([z, s], dim=1)
            
        elif random_flag == 2:
            s_mean = self.m_hidden2mean_s(last_en_hidden)
            s_logv = self.m_hidden2logv_s(last_en_hidden)
            s_std = torch.exp(0.5*s_logv)
            s = torch.randn_like(s_std)*s_std + s_mean

            s_prior = self.m_item_embedding(item_ids)

            l_mean = self.m_hidden2mean_l(last_en_hidden)
            l_logv = self.m_hidden2logv_l(last_en_hidden)
            l_std = torch.exp(0.5*l_logv)
            l = torch.randn_like(l_std)*l_std + l_mean

            # variational_hidden = torch.cat([s, l], dim=1)
            variational_hidden = s+l

        elif random_flag == 3:
            z_mean = self.m_hidden2mean_z(last_en_hidden)
            z_logv = self.m_hidden2logv_z(last_en_hidden)
            z_std = torch.exp(0.5*z_logv)
            z = torch.randn_like(z_std)*z_std + z_mean

            z_prior = self.m_user_embedding(user_ids)

            l_mean = self.m_hidden2mean_l(last_en_hidden)
            l_logv = self.m_hidden2logv_l(last_en_hidden)
            l_std = torch.exp(0.5*l_logv)
            l = torch.randn_like(l_std)*l_std + l_mean

            variational_hidden = z+l
            # variational_hidden = torch.cat([z, l], dim=1)
        else:
            raise NotImplementedError("0, 1, 2, 3, variational not defined!")

        # init_de_hidden = self.m_latent2hidden(variational_hidden)

        # repeat_init_de_hidden = init_de_hidden.unsqueeze(1)
        # repeat_init_de_hidden = repeat_init_de_hidden.expand(init_de_hidden.size(0), input_de_embedding.size(1), init_de_hidden.size(-1))

        # input_de_embedding = input_de_embedding+repeat_init_de_hidden

        hidden = None

        de_batch_size = input_de_sequence.size(0)
        de_len = input_de_sequence.size(1)
        # print("decoding length", de_len)

        output = []
        var_de = self.m_latent2hidden(variational_hidden)

        input_de_embedding = self.m_embedding(input_de_sequence)
        input_de_embedding_attn = input_de_embedding.transpose(0, 1)

        input_de_embedding_attn = input_de_embedding_attn*math.sqrt(self.m_embedding_size)
        input_de_embedding_attn = self.m_pos_encoder(input_de_embedding_attn)

        if self.m_de_mask is None or self.m_de_mask.size(0) != input_de_sequence.size(1):
            self.m_de_mask = self.f_generate_square_subsequent_mask(input_de_sequence.size(1)).to(self.m_device)

        decode_strategy = self.m_decode_strategy

        if decode_strategy == "avg":
            """
            avg mechanism
            """

            for de_step_i in range(de_len):
                input_de_step_i = input_de_embedding[:, de_step_i, :]
                input_de_step_i = input_de_step_i.unsqueeze(1)
                # output_step_i, hidden = self.m_decoder_rnn(input_de_step_i, hidden)
                
                output.append(output_step_i)

                # var_de_flag = self.m_decoder_gate(output_step_i.squeeze(1))
                # # print("var_de_flag", var_de_flag.size())
                # var_de = self.m_latent2hidden((1-var_de_flag)*z+var_de_flag*s+l)
            output = torch.cat(output, dim=1)

        elif decode_strategy == "gating":
            """
            gating mechanism
            """

            for de_step_i in range(de_len):
                input_de_step_i = input_de_embedding[:, de_step_i, :]+var_de
                input_de_step_i = input_de_step_i.unsqueeze(1)
                # output_step_i, hidden = self.m_decoder_rnn(input_de_step_i, hidden)
                # output.append(output_step_i)

                var_de_flag = self.m_decoder_gate(output_step_i.squeeze(1))
                # print("var_de_flag", var_de_flag.size())
                var_de = self.m_latent2hidden((1-var_de_flag)*z+var_de_flag*s+l)

                output_step_i = output_step_i+var_de

            output = torch.cat(output, dim=1)

        elif decode_strategy == "attn":
            """
            attention mechanism output
            """

            ### var_de: batch_size*3*latent_size

            z_de = self.m_latent2hidden(z)
            s_de = self.m_latent2hidden(s)
            l_de = self.m_latent2hidden(l)

            var_de = torch.cat([z_de.unsqueeze(1), s_de.unsqueeze(1), l_de.unsqueeze(1)], dim=1)
            # print("var de size", var_de.size())
            
            ### var_de: 3*batch_size*latent_size
            var_de = var_de.transpose(0, 1)
            
            ### input_de_embedding_attn: len*batch_size*latent_size
            # input_de_embedding_attn = input_de_embedding.transpose(0, 1)

            # print("input_de_embedding_attn size", input_de_embedding_attn.size())
            # print("var_de size", var_de.size())
            # print("mask size", self.m_de_mask.size())

            output = self.m_transformer_decoder(input_de_embedding_attn, var_de, tgt_mask=self.m_de_mask)
            output = output.transpose(0, 1)

        output = output.contiguous()
        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        
        return logits, z_prior, z_mean, z_logv, z, s_prior, s_mean, s_logv, s, l_mean, l_logv, l, variational_hidden    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)
