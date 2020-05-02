import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class REVIEWDI(nn.Module):
    def __init__(self, vocab_obj, args, device):
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
        self.m_user_size = vocab_obj.user_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_user_embedding_size)

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=True)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)

        self.m_hidden_factor = (2 if self.m_bidirectional else 1)*self.m_num_layers

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        self.m_hidden2mean_s = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_s = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        # self.m_latent2hidden = nn.Linear(self.m_latent_size*2, self.m_hidden_size)
        self.m_latent2hidden = nn.Linear(self.m_latent_size*2, self.m_embedding_size)
        # self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)
        self.m_output2vocab = nn.Linear(self.m_hidden_size*(2 if self.m_bidirectional else 1), self.m_vocab_size)

        self.m_latent2RRe = nn.Linear(self.m_latent_size, self.m_hidden_size)
        self.m_latent2ARe = nn.Linear(self.m_latent_size, self.m_hidden_size)
        # self.m_latent2RRe = nn.Linear(self.m_latent_size, self.m_vocab_size)
        # self.m_latent2ARe = nn.Linear(self.m_latent_size, self.m_vocab_size)

        self = self.to(self.m_device)

    def forward(self, input_sequence, user_ids, length):
        batch_size = input_sequence.size(0)

        input_embedding = self.m_embedding(input_sequence)
        input_embedding = self.m_embedding_dropout(input_embedding)
        encoder_outputs, _ = self.m_encoder_rnn(input_embedding)

        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (length-1).long()
        
        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        z_mean = self.m_hidden2mean_z(last_en_hidden)
        z_logv = self.m_hidden2logv_z(last_en_hidden)
        z_std = torch.exp(0.5*z_logv)
        z = torch.randn_like(z_std)*z_std + z_mean

        s_mean = self.m_hidden2mean_s(last_en_hidden)
        s_logv = self.m_hidden2logv_s(last_en_hidden)
        s_std = torch.exp(0.5*s_logv)
        s = torch.randn_like(s_std)*s_std + s_mean

        # z_s = torch.cat([z, s], dim=1)
        ### z: RRe
        z_s = z
        # z_s = z 

        init_en_hidden = self.m_latent2hidden(z_s)

        input_embedding = self.m_embedding(input_sequence)
        repeat_init_en_hidden = init_en_hidden.unsqueeze(1)
        repeat_init_en_hidden = repeat_init_en_hidden.expand(init_en_hidden.size(0), input_embedding.size(1), init_en_hidden.size(-1))

        input_embedding = input_embedding+repeat_init_en_hidden

        hidden = None
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)
        # if self.m_word_dropout_rate > 0:
        #     prob = torch.rand(input_sequence.size()).to(self.m_device)

        #     prob[(input_sequence.data-self.m_sos_idx)*(input_sequence.data-self.m_pad_idx) ==0] = 1

        #     decoder_input_sequence = decoder_input_sequence.clone()
        #     decoder_input_sequence = input_sequence.clone()

        #     decoder_input_sequence[prob < self.m_word_dropout_rate] = self.m_unk_idx
        #     input_embedding = self.m_embedding(decoder_input_sequence)

        ### Rec loss
        ### output: batch_size*seq_len*hidden_size
        ### logits: (batch_size*seq_len)*hidden_size
        output = output.contiguous()
        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        # logp = nn.functional.log_softmax(, dim=-1)
        # logp = logp.view(batch_size, -1, self.m_vocab_size)

        ### ARe loss

        ### ARe_logits: batch_size*hidden_size
        ARe_logits = self.m_latent2ARe(s)

        ### ARe_pred: batch_size*voc_size
        ARe_pred = self.m_output2vocab(ARe_logits.view(-1, ARe_logits.size(-1)))

        # ARe_pred = F.softmax(ARe_pred, dim=1)
        ### RRe loss

        RRe_logits = self.m_latent2RRe(z)
        RRe_pred = self.m_output2vocab(RRe_logits.view(-1, RRe_logits.size(-1)))
       
        # RRe_pred = self.m_latent2RRe(z)
        
        # RRe_pred = F.softmax(RRe_pred, dim=1)

        return logits, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred

        

            
