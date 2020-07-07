import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class _ENC_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args):
        super(_ENC_NETWORK, self).__init__()

        # self.m_device = device
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_hidden_size = args.hidden_size
        self.m_layers_num = args.layers_num
        self.m_dropout_rate = args.dropout
        self.m_latent_size = args.latent_size

        self.m_embedding_size = args.embedding_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_size, self.m_latent_size)

        user_cnt = torch.zeros((self.m_user_size, 1))
        item_cnt = torch.zeros((self.m_item_size, 1))

        self.register_buffer("m_user_cnt", user_cnt)
        self.register_buffer("m_item_cnt", item_cnt)

        self.m_user_encoder = _ENCODER(self.m_embedding, self.m_latent_size, self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)
        self.m_item_encoder = _ENCODER(self.m_embedding, self.m_latent_size, self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)

        self.m_user_decoder = _DECODER(self.m_embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)
        self.m_item_decoder = _DECODER(self.m_embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        # self = self.to(self.m_device)

    def forward(self, reviews, review_lens, user_ids, item_ids):

        ### obtain user representation
        user_hidden = self.m_user_encoder(reviews, review_lens)

        ### obtain item representation
        item_hidden = self.m_item_encoder(reviews, review_lens)

        user_output = self.m_user_decoder(reviews, user_hidden)
        item_output = self.m_item_decoder(reviews, item_hidden)

        user_logits = self.m_output2vocab(user_output.view(-1, user_output.size(2)))
        item_logits = self.m_output2vocab(item_output.view(-1, item_output.size(2)))

        return user_logits, item_logits

    def update_user_item(self, reviews, review_lens, user_ids, item_ids):
        
        user_hidden = self.m_user_encoder(reviews, review_lens)
        user_one_hot = F.one_hot(user_ids, self.m_user_size).type(user_hidden.dtype)
        user_embedding_sum = user_one_hot.transpose(0, 1) @ user_hidden

        self.m_user_embedding.weight.data.add_(user_embedding_sum)

        self.m_user_cnt.add_(torch.sum(user_one_hot, dim=0).unsqueeze(1))

        item_hidden = self.m_item_encoder(reviews, review_lens)
        item_one_hot = F.one_hot(item_ids, self.m_item_size).type(item_hidden.dtype)
        item_embedding_sum = item_one_hot.transpose(0, 1) @ item_hidden

        self.m_item_embedding.weight.data.add_(item_embedding_sum)

        self.m_item_cnt.add_(torch.sum(item_one_hot, dim=0).unsqueeze(1))

    def normalize_user_item(self):
        self.m_user_embedding.weight.data.div_(self.m_user_cnt)
        self.m_item_embedding.weight.data.div_(self.m_item_cnt)
        # self.m_item_embedding.weight.data = self.m_item_embedding.weight.data/self.m_item_cnt

        if (self.m_user_cnt != 0).sum() != self.m_user_size:
            print("user num", (self.m_user_cnt == 0).sum())
            print("user num", self.m_user_size)
            for i, _ in enumerate(self.m_user_cnt):
                if self.m_user_cnt[i, 0] == 0:
                    print("user cnt zeros", self.m_user_cnt[i, 0])

        if torch.isinf(self.m_user_embedding.weight).any():
            print("normalize user_embedding inf", self.m_user_embedding.weight)
        
        if torch.isinf(self.m_item_embedding.weight).any():
            print("normalize item_embedding inf", self.m_item_embedding.weight)

        self.m_user_embedding.weight.requires_grad=False
        self.m_item_embedding.weight.requires_grad=False
        self.m_embedding.weight.requires_grad=False
        self.m_output2vocab.weight.requires_grad=False

class _ENCODER(nn.Module):
    def __init__(self, embedding, latent_size, hidden_size, layers_num=1, dropout=0.3):
        super(_ENCODER, self).__init__()
        
        self.m_dropout_rate = dropout
        # self.m_device = device

        self.m_embedding = embedding
        self.m_embedding_dropout = nn.Dropout(self.m_dropout_rate)

        self.m_latent_size = latent_size
        self.m_hidden_size = hidden_size
        self.m_layers_num = layers_num

        self.m_gru = nn.GRU(self.m_hidden_size, self.m_hidden_size, self.m_layers_num, dropout=self.m_dropout_rate, bidirectional=True)
        
        self.m_hidden2latent = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        
        # self = self.to(self.m_device)

    def forward(self, x, x_len, hidden=None):
        # if not hasattr(self, '_flattened'):
        #     self.m_gru.flatten_parameters()
        #     setattr(self, '_flattened', True)

        batch_size = x.size(0)
        input_embedding = self.m_embedding(x)
        input_embedding = self.m_embedding_dropout(input_embedding)

        encoder_outputs, _ = self.m_gru(input_embedding)

        # first_dim_index = torch.arange(batch_size).to(self.m_device)
        first_dim_index = torch.arange(batch_size).to(input_embedding.device)
        second_dim_index = (x_len-1).long()

        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        en_latent = self.m_hidden2latent(last_en_hidden)

        return en_latent

class _DECODER(nn.Module):
    def __init__(self, embedding, embedding_size, latent_size, hidden_size, layers_num=1, dropout=0.3):
        super(_DECODER, self).__init__()
        # self.m_device = device

        self.m_latent_size = latent_size
        self.m_dropout_rate = dropout
        self.m_embedding_size = embedding_size
        
        self.m_tanh = nn.Tanh()
        self.m_latent2output = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_embedding = embedding
        self.m_embedding_dropout = nn.Dropout(self.m_dropout_rate)

        self.m_hidden_size = hidden_size
        self.m_layers_num = layers_num
        
        self.m_gru = nn.GRU(self.m_hidden_size, self.m_hidden_size, self.m_layers_num, dropout=self.m_dropout_rate, bidirectional=False)

        # self = self.to(self.m_device)

    def forward(self, x, en_latent, hidden=None):
        # if not hasattr(self, '_flattened'):
        #     self.m_gru.flatten_parameters()
        #     setattr(self, '_flattened', True)

        en_hidden = self.m_tanh(en_latent)
        de_hidden = self.m_latent2output(en_hidden)

        batch_size = x.size(0)
        input_embedding = self.m_embedding(x)
        input_embedding = self.m_embedding_dropout(input_embedding)

        de_hidden = de_hidden.unsqueeze(1)
        de_hidden = de_hidden.expand(de_hidden.size(0), input_embedding.size(1), de_hidden.size(-1))

        output_embedding = input_embedding + de_hidden

        output, hidden = self.m_gru(output_embedding)
        output = output.contiguous()

        return output

class _GEN_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args):
        super().__init__()

        # self.m_device = device

        self.m_hidden_size = args.hidden_size
        self.m_latent_size = args.latent_size
        
        self.m_max_sequence_len = args.max_seq_length
        self.m_layers_num = args.layers_num
        self.m_bidirectional = args.bidirectional
        self.m_embedding_size = args.embedding_size
        self.m_aspect_num = args.aspect_num

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_unk_idx = vocab_obj.unk_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_size
        self.m_item_num = vocab_obj.item_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_latent_size)

        self.m_aspect_embedding = nn.Linear(self.m_embedding_size, self.m_aspect_num, bias=False)

        self.m_tanh = nn.Tanh()
        self.m_latent2output = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_de_strategy = args.de_strategy

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_layers_num, bidirectional=False, batch_first=True)

        self.m_gating_network = nn.Sequential(nn.Linear(self.m_hidden_size, self.m_embedding_size), nn.Tanh(), nn.Linear(self.m_embedding_size, 2))
        
        self.m_user_aspect = nn.Linear(self.m_latent_size, self.m_embedding_size, bias=False)
        self.m_item_aspect = nn.Linear(self.m_latent_size, self.m_embedding_size, bias=False)
            
        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        if self.m_de_strategy == "attn":
            self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

        # self = self.to(self.m_device)
        
    def f_init_user_item_word(self, E_network):
        E_network.eval()

        self.m_embedding.weight.requires_grad=False
        self.m_user_embedding.weight.requires_grad=False
        self.m_item_embedding.weight.requires_grad=False
        self.m_output2vocab.weight.requires_grad=False

        self.m_embedding.weight.data.copy_(E_network.m_embedding.weight.data)

        self.m_user_embedding.weight.data.copy_(E_network.m_user_embedding.weight.data)
        
        self.m_item_embedding.weight.data.copy_(E_network.m_item_embedding.weight.data)
       
        self.m_output2vocab.weight.data.copy_(E_network.m_output2vocab.weight.data)
        
    def f_gumbel_softmax(self, logits, eps=1e-20, temp=1e-3):
        shape = logits.size()
        U = torch.rand(shape)
        gumbel_sample = -torch.log(-torch.log(U+eps)+eps).to(logits.device)
        y = logits + gumbel_sample
    
        return F.softmax(y/temp, dim=-1)

    def forward(self, input_de_sequence, user_ids, item_ids, random_flag):

        batch_size = input_de_sequence.size(0)
        de_batch_size = input_de_sequence.size(0)
        de_len = input_de_sequence.size(1)

        input_de_embedding = self.m_embedding(input_de_sequence)

        input_user_hidden = self.m_user_embedding(user_ids)
        input_item_hidden = self.m_item_embedding(item_ids)

        ### m_aspect_word_embedding: voc_size*aspect_size
        self.m_aspect_word_embedding = F.softmax(self.m_aspect_embedding(self.m_embedding.weight), dim=0)

        ### m_aspect_word_embedding: aspect_size*voc_size
        self.m_aspect_word_embedding = self.m_aspect_word_embedding.transpose(0, 1)

        ### aspect_prop: batch_size*aspect_size
        aspect_prop = self.m_aspect_embedding(self.m_user_aspect(input_user_hidden)+self.m_item_aspect(input_item_hidden))

        # print("aspect_prop", aspect_prop.size())
        # print("aspect_prop device", aspect_prop.device)
        # print("self.m_aspect_word_embedding device", self.m_aspect_word_embedding.device)
        # print("m_aspect_word_embedding", self.m_aspect_word_embedding.size())
        ### aspect_word_prob: batch_size*voc_size
        # aspect_word_prob = F.softmax(aspect_prop @ self.m_aspect_word_embedding, dim=-1)
        aspect_word_prob = aspect_prop @ self.m_aspect_word_embedding
        if torch.isnan(aspect_word_prob).any():
            print("aspect_word_prob", aspect_word_prob)
            exit()

        output = []
        # user_item_hidden_i = self.m_latent2output(user_item_hidden_init)

        hidden = None
        decode_strategy = self.m_de_strategy

        if decode_strategy == "attn":
            """
            attention mechanism output
            """

            for de_step_i in range(de_len):
                input_de_step_i = input_de_embedding[:, de_step_i, :]
                # input_de_step_i = input_de_step_i + user_item_hidden_i
                input_de_step_i = input_de_step_i.unsqueeze(1)
                
                output_step_i, hidden = self.m_decoder_rnn(input_de_step_i, hidden)
                
                ### output_step_i: batch_size*hidden_size
                output_step_i = output_step_i.squeeze(1)

                ### word_prob: batch_size*voc_size
                # word_prob_i = F.softmax(self.m_output2vocab(output_step_i), dim=-1)
                word_prob_i = self.m_output2vocab(output_step_i)
                if torch.isnan(word_prob_i).any():
                    print("word_prob_i", word_prob_i)
                    exit()
                
                ### gate: batch_size*2
                gate_i = self.m_gating_network(output_step_i)

                gate_i = self.f_gumbel_softmax(gate_i) 

                # gate_i = F.softmax(gate_i)

                ### gate_aspect_word_prob: batch_size*voc_size
                gate_aspect_word_prob_i = gate_i[:, 0].unsqueeze(1)*aspect_word_prob

                ### gate_word_prob: batch_size*voc_size
                gate_word_prob_i = gate_i[:, 1].unsqueeze(1)*word_prob_i

                output_prob_i = gate_aspect_word_prob_i + gate_word_prob_i

                output.append(output_prob_i.unsqueeze(1))

            output = torch.cat(output, dim=1)

        output = output.contiguous()
        
        if torch.isnan(output).any():
            print("output", output)
            exit()

        return output

### obtain the representation of users and items. 

