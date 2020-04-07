import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class REVIEWDI(nn.Module):
    def __init__(self, vocab_obj, user_num, args, device):
        super().__init__()

        self.m_device=device

        self.m_embedding_size = args.embedding_size
        self.m_user_embedding_size = args.latent_size

        self.m_hidden_size = args.hidden_size
        self.m_word_dropout_rate = args.word_dropout
        self.m_embedding_dropout = args.embedding_dropout
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
        self.m_user_size = user_num

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_user_embedding_size)

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)

        self.m_hidden_factor = (2 if self.m_bidirectional else 1)*self.m_num_layers

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)

        self.m_hidden2mean_s = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)
        self.m_hidden2logv_s = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)

        self.m_latent2hidden = nn.Linear(self.m_latent_size*2, self.m_hidden_size*self.m_hidden_factor)
        self.m_output2vocab = nn.Linear(self.m_hidden_size*(2 if self.m_bidirectional else 1), self.m_vocab_size)

        self.m_latent2RRe = nn.Linear(self.m_latent_size, self.m_vocab_size)
        self.m_latent2ARe = nn.Linear(self.m_latent_size, self.m_vocab_size)

        self = self.to(self.m_device)

    def forward(self, input_sequence, user_ids, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        input_embedding = self.m_embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.m_encoder_rnn(packed_input)

        if self.m_bidirectional or self.m_num_layers > 1:
            hidden = hidden.view(batch_size, self.m_hidden_size*self.m_hidden_factor)
        else:
            hidden = hidden.squeeze()

        z_mean = self.m_hidden2mean_z(hidden)
        z_logv = self.m_hidden2logv_z(hidden)
        z_std = torch.exp(0.5*z_logv)

        user_embedding = self.m_user_embedding(user_ids)
        z_mean_prior = user_embedding
        # z_logv_prior

        # print("device", self.m_device)
        pre_z = torch.randn([batch_size, self.m_latent_size]).to(self.m_device)
        z = pre_z*z_std + z_mean

        s_mean = self.m_hidden2mean_s(hidden)
        s_logv = self.m_hidden2logv_s(hidden)
        s_std = torch.exp(0.5*s_logv)

        pre_s = torch.randn([batch_size, self.m_latent_size]).to(self.m_device)
        s = pre_s*s_std + s_mean

        z_s = torch.cat([z, s], dim=1)

        hidden = self.m_latent2hidden(z_s)

        if self.m_bidirectional or self.m_num_layers > 1:
            hidden = hidden.view(self.m_hidden_factor, batch_size, self.m_hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        if self.m_word_dropout_rate > 0:
            prob = torch.rand(input_sequence.size()).to(self.m_device)

            prob[(input_sequence.data-self.m_sos_idx)*(input_sequence.data-self.m_pad_idx) ==0] = 1

            decoder_input_sequence = decoder_input_sequence.clone()
            decoder_input_sequence = input_sequence.clone()

            decoder_input_sequence[prob < self.m_word_dropout_rate] = self.m_unk_idx
            input_embedding = self.m_embedding(decoder_input_sequence)

        input_embedding = self.m_embedding_dropout(input_embedding)
        
        ### concatenate hidden with input 
        repeat_hidden = hidden.squeeze(0)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        outputs, _ = self.m_decoder_rnn(packed_input, hidden)

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        # batch_size, , _ = padded_outputs.size()

        logp = nn.functional.log_softmax(self.m_output2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(batch_size, -1, self.m_vocab_size)

        ### ARe loss

        ARe_pred = self.m_latent2ARe(s)

        # ARe_pred = F.softmax(ARe_pred, dim=1)
        ### RRe loss

        RRe_pred = self.m_latent2RRe(z)
        
        # RRe_pred = F.softmax(RRe_pred, dim=1)

        return logp, z_mean_prior, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred

        

            
