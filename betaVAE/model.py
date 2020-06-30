import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class _NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device=device

        self.m_embedding_size = args.embedding_size

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

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=True)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)
        # self.m_decoder_rnn = nn.GRU(self.m_embedding_size*2, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)

        self.m_hidden_factor = (2 if self.m_bidirectional else 1)*self.m_num_layers

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        # self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_hidden_size*self.m_hidden_factor)
        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)
        # self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_hidden_size)

        self.m_output2vocab = nn.Linear(self.m_hidden_size*(2 if self.m_bidirectional else 1), self.m_vocab_size)

        self = self.to(self.m_device)

    def f_encode(self, input_sequence, length):
        batch_size = input_sequence.size(0)

        input_embedding = self.m_embedding(input_sequence)
        input_embedding = self.m_embedding_dropout(input_embedding)

        en_outputs, _ = self.m_encoder_rnn(input_embedding)

        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (length-1).long()

        ### batch_size*hidden_size
        last_en_hidden = en_outputs[first_dim_index, second_dim_index, :].contiguous()

        if self.m_bidirectional or self.m_num_layers > 1:
            last_en_hidden = last_en_hidden.view(batch_size, self.m_hidden_size*self.m_hidden_factor)
        else:
            last_en_hidden = last_en_hidden.squeeze()

        en_seq = last_en_hidden
        return en_seq

    def f_reparameterize(self, en_seq):
        
        z_mean = self.m_hidden2mean_z(en_seq)
        z_logv = self.m_hidden2logv_z(en_seq)
        z_std = torch.exp(0.5*z_logv)
    
        pre_z = torch.randn_like(z_std)

        z = pre_z*z_std + z_mean

        return z_mean, z_logv, z

    def f_decode(self, var_seq, input_sequence):
        input_embedding = self.m_embedding(input_sequence)
        input_embedding = self.m_embedding_dropout(input_embedding)

        var_seq_de = var_seq.unsqueeze(1)
        var_seq_de = var_seq_de.expand(var_seq_de.size(0), input_embedding.size(1), var_seq_de.size(-1))

        input_embedding = input_embedding+var_seq_de

        hidden = None
        # hidden = init_de_hidden.unsqueeze(0)
        # print("input_embedding", input_embedding)
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)
        # print("output", output)
        output = output.contiguous()

        return output

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        en_seq = self.f_encode(input_sequence, length)

        z_mean, z_logv, z = self.f_reparameterize(en_seq)

        if len(z.size()) < 2:
            z = z.unsqueeze(0)

        # print()
        var_seq = self.m_latent2hidden(z)

        output = self.f_decode(var_seq, input_sequence)

        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        # print("logits", logits)

        logp = F.log_softmax(logits, dim=-1)
        logp = logp.view(batch_size, -1, self.m_vocab_size)

        return logp, z_mean, z_logv, z, en_seq, var_seq

        

            
