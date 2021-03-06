import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class REVIEWDI(nn.Module):
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

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)

        self.m_hidden_factor = (2 if self.m_bidirectional else 1)*self.m_num_layers

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*self.m_hidden_factor, self.m_latent_size)

        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_hidden_size*self.m_hidden_factor)

        # self.m_hidden2hidden = nn.Linear(self.m_hidden_size, self.m_hidden_size)
        self.m_hidden2hidden = nn.Linear(self.m_hidden_size, self.m_embedding_size)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        self = self.to(self.m_device)

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        input_embedding = self.m_embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        encoder_outputs, hidden = self.m_encoder_rnn(packed_input)

        encoder_outputs = rnn_utils.pad_packed_sequence(encoder_outputs, batch_first=True)[0]
        encoder_outputs = encoder_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        encoder_outputs = encoder_outputs[reversed_idx]

        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (length-1).long()

        ### batch_size*hidden_size
        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        if self.m_bidirectional or self.m_num_layers > 1:
            last_en_hidden = last_en_hidden.view(batch_size, self.m_hidden_size*self.m_hidden_factor)
        else:
            last_en_hidden = last_en_hidden.squeeze()

        hidden_0 = self.m_hidden2hidden(last_en_hidden)

        init_de_hidden = torch.zeros([batch_size, self.m_hidden_size]).to(self.m_device)

        if self.m_bidirectional or self.m_num_layers > 1:
            init_de_hidden = init_de_hidden.view(self.m_hidden_factor, batch_size, self.m_hidden_size)
        else:
            init_de_hidden = init_de_hidden.unsqueeze(0)

        if self.m_word_dropout_rate > 0:
            prob = torch.rand(input_sequence.size()).to(self.m_device)

            prob[(input_sequence.data-self.m_sos_idx)*(input_sequence.data-self.m_pad_idx) ==0] = 1

            decoder_input_sequence = input_sequence.clone()

            decoder_input_sequence[prob < self.m_word_dropout_rate] = self.m_unk_idx
            input_embedding = self.m_embedding(decoder_input_sequence)

        input_embedding = self.m_embedding_dropout(input_embedding)
        
        ### concatenate hidden with input 
        repeat_hidden_0 = hidden_0.unsqueeze(1)
        repeat_hidden_0 = repeat_hidden_0.expand(hidden_0.size(0), input_embedding.size(1), hidden_0.size(-1))

        input_embedding = input_embedding+repeat_hidden_0
        # input_embedding = torch.cat([input_embedding, repeat_hidden_0], dim=-1)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        outputs, _ = self.m_decoder_rnn(packed_input)

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()

        _, reversed_idx = torch.sort(sorted_idx)
        output = padded_outputs[reversed_idx]

        # output = torch.cat([padded_outputs, repeat_hidden_0], dim=-1)

        logp = nn.functional.log_softmax(self.m_output2vocab(output.view(-1, output.size(2))), dim=-1)
        logp = logp.view(batch_size, -1, self.m_vocab_size)

        return logp, _, _, _, last_en_hidden , hidden_0
