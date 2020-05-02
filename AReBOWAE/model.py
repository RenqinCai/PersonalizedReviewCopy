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
        # self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)

        # self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        # self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        self.m_hidden2mean_s = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_s = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        # self.m_latent2hidden = nn.Linear(self.m_latent_size*2, self.m_hidden_size)
        # self.m_latent2hidden = nn.Linear(self.m_latent_size*2, self.m_embedding_size)
        # self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)
        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        # self.m_latent2RRe = nn.Linear(self.m_latent_size, self.m_hidden_size)
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

        s_mean = self.m_hidden2mean_s(last_en_hidden)
        s_logv = self.m_hidden2logv_s(last_en_hidden)
        s_std = torch.exp(0.5*s_logv)
        s = torch.randn_like(s_std)*s_std + s_mean

        ### ARe_logits: batch_size*hidden_size
        ARe_hidden = self.m_latent2ARe(s)

        ### ARe_pred: batch_size*voc_size
        ARe_logits = self.m_output2vocab(ARe_hidden.view(-1, ARe_hidden.size(-1)))

        return ARe_logits, s_mean, s_logv, s

        

            
