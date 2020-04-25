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

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=True)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)
        # self.m_decoder_rnn = nn.GRU(self.m_embedding_size*2, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=self.m_bidirectional, batch_first=True)

        self.m_hidden_factor = (2 if self.m_bidirectional else 1)*self.m_num_layers

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        # self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_hidden_size*self.m_hidden_factor)
        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_output2vocab = nn.Linear(self.m_hidden_size*(2 if self.m_bidirectional else 1), self.m_vocab_size)

        self = self.to(self.m_device)

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        
        sorted_input_sequence = input_sequence[sorted_idx]

        input_embedding = self.m_embedding(sorted_input_sequence)
        
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

        z_mean = self.m_hidden2mean_z(last_en_hidden)
        z_logv = self.m_hidden2logv_z(last_en_hidden)
        z_std = torch.exp(0.5*z_logv)

        # print("z_mean", z_mean)
        # pre_z = torch.randn([batch_size, self.m_latent_size]).to(self.m_device)
        pre_z = torch.randn_like(z_std)
        # print("pre_z", pre_z)
        # exit()
        # print("z_std", z_std)
        # print("pre_z*z_std", pre_z*z_std)
        z = pre_z*z_std + z_mean
        # print("z", z)
        
        if len(z.size()) < 2:
            z = z.unsqueeze(0)

        # print()
        init_de_hidden = self.m_latent2hidden(z)
        # print("init_de_hidden", init_de_hidden)    
        
        # if self.m_bidirectional or self.m_num_layers > 1:
        #     init_de_hidden = init_de_hidden.view(self.m_hidden_factor, batch_size, self.m_hidden_size)
        # else:
        #     init_de_hidden = init_de_hidden.unsqueeze(0)

        # if self.m_word_dropout_rate > 0:
        #     prob = torch.rand(input_sequence.size()).to(self.m_device)

        #     prob[(input_sequence.data-self.m_sos_idx)*(input_sequence.data-self.m_pad_idx) ==0] = 1

        #     decoder_input_sequence = input_sequence.clone()

        #     decoder_input_sequence[prob < self.m_word_dropout_rate] = self.m_unk_idx
        #     input_embedding = self.m_embedding(decoder_input_sequence)

        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        
        # input_sequence = input_sequence[sorted_idx]
        # print("input", input_sequence)

        input_embedding = self.m_embedding(input_sequence)
        input_embedding = self.m_embedding_dropout(input_embedding)

        repeat_hidden_0 = init_de_hidden.unsqueeze(1)
        repeat_hidden_0 = repeat_hidden_0.expand(init_de_hidden.size(0), input_embedding.size(1), init_de_hidden.size(-1))

        # print("0 input_embedding", input_embedding)
        # print("repeat_hidden_0", repeat_hidden_0)

        input_embedding = input_embedding+repeat_hidden_0
        # input_embedding = torch.cat([input_embedding, repeat_hidden_0], dim=-1)

        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        # outputs, _ = self.m_decoder_rnn(packed_input, init_de_hidden)
        # outputs, _ = self.m_decoder_rnn(packed_input)

        # padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        # padded_outputs = padded_outputs.contiguous()

        # _, reversed_idx = torch.sort(sorted_idx)
        # output = padded_outputs[reversed_idx]

        hidden = None
        # print("input_embedding", input_embedding)
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)
        # print("output", output)
        output = output.contiguous()

        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        # print("logits", logits)

        logp = nn.functional.log_softmax(logits, dim=-1)
        logp = logp.view(batch_size, -1, self.m_vocab_size)

        return logp, z_mean, z_logv, z, init_de_hidden, last_en_hidden

        

            
