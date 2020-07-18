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
        self.m_cont_vocab_size = vocab_obj.m_cont_vocab_size

        print("self.m_cont_vocab_size: ", self.m_cont_vocab_size)
        print("self.m_vocab_size: ", self.m_vocab_size)

        self.m_hidden_size = args.hidden_size
        self.m_layers_num = args.layers_num
        self.m_dropout_rate = args.dropout
        self.m_latent_size = args.latent_size

        self.m_embedding_size = args.embedding_size

        # self.m_embedding = nn.Embedding(self.m_cont_vocab_size, self.m_embedding_size)
        self.m_input_layer = nn.Linear(self.m_cont_vocab_size, self.m_embedding_size)
        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_size, self.m_latent_size)

        user_cnt = torch.zeros((self.m_user_size, 1))
        item_cnt = torch.zeros((self.m_item_size, 1))

        self.register_buffer("m_user_cnt", user_cnt)
        self.register_buffer("m_item_cnt", item_cnt)

        self.m_user_encoder = _ENCODER(self.m_input_layer, self.m_embedding_size, self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)
        self.m_item_encoder = _ENCODER(self.m_input_layer, self.m_embedding_size,self.m_hidden_size, self.m_layers_num, self.m_dropout_rate)

        ### hidden_size*voc_size
        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_cont_vocab_size)

        # self = self.to(self.m_device)

    def forward(self, reviews, user_ids, item_ids):

        ### obtain user representation

        user_hidden = self.m_user_encoder(reviews)

        ### obtain item representation
        item_hidden = self.m_item_encoder(reviews)

        # print("forward ...")

        user_logits = self.m_output2vocab(user_hidden)
        item_logits = self.m_output2vocab(item_hidden)

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

        self.m_user_embedding.weight.requires_grad=False
        self.m_item_embedding.weight.requires_grad=False
        self.m_embedding.weight.requires_grad=False
        self.m_output2vocab.weight.requires_grad=False

class _ENCODER(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, layers_num=1, dropout=0.3):
        super(_ENCODER, self).__init__()
        
        self.m_dropout_rate = dropout
        # self.m_device = device

        self.m_input_layer = embedding
        self.m_embedding_dropout = nn.Dropout(self.m_dropout_rate)

        self.m_embedding_size = embedding_size
        self.m_hidden_size = hidden_size
        self.m_layers_num = layers_num

        self.m_input_fc = nn.Linear(self.m_embedding_size, self.m_hidden_size)

        self.m_input_af = nn.Tanh()

        # self = self.to(self.m_device)

    def forward(self, x, hidden=None):

        # print("x size: ", x.size())

        ### x: batch_size*voc_size
        batch_size = x.size(0)

        ### input_embedding: batch_size*embedding_size
        input_embedding = self.m_input_layer(x)

        # print("input_embedding size: ", input_embedding.size())

        ### input_embedding: batch_size*embedding_size
        input_embedding = self.m_embedding_dropout(input_embedding)
        hidden = self.m_input_af(input_embedding)

        ### hidden: batch_size*hidden_size
        hidden = self.m_input_fc(hidden)
        hidden = self.m_input_af(hidden)

        # print("hidden size: ", hidden.size())

        return hidden

class _GEN_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args):
        super().__init__()

        # self.m_device = device

        self.m_embedding_size = args.embedding_size
        self.m_hidden_size = args.hidden_size
        self.m_latent_size = args.latent_size
        
        self.m_max_sequence_len = args.max_seq_length
        self.m_layers_num = args.layers_num
        self.m_bidirectional = args.bidirectional
        
        self.m_word_type_num = args.word_type_num
        self.m_l_embedding_size = args.l_embedding_size
        self.m_l_hidden_size = args.l_hidden_size
        
        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_unk_idx = vocab_obj.unk_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_size
        self.m_item_num = vocab_obj.item_size

        self.m_func_vocab_size = vocab_obj.m_func_vocab_size
        self.m_cont_vocab_size = vocab_obj.m_cont_vocab_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_latent_size)
        self.m_bow_input_layer = nn.Linear(self.m_cont_vocab_size, self.m_embedding_size)

        self.m_l_embedding = nn.Embedding(self.m_word_type_num, self.m_l_embedding_size)

        self.m_mu_p_fc = nn.Linear(self.m_embedding_size, self.m_latent_size)
        self.m_logvar_p_fc = nn.Linear(self.m_embedding_size, self.m_latent_size)

        self.m_mu_fc = nn.Linear(self.m_embedding_size, self.m_latent_size)
        self.m_logvar_fc = nn.Linear(self.m_embedding_size, self.m_latent_size)

        self.m_tanh = nn.Tanh()
        self.m_latent2embed = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_de_strategy = args.de_strategy

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_layers_num, bidirectional=False, batch_first=True)
        
        self.m_decoder_l_rnn = nn.GRU(self.m_l_embedding_size+self.m_embedding_size, self.m_l_hidden_size, num_layers=self.m_layers_num, bidirectional=False, batch_first=True)

        self.m_output2funcvocab = nn.Linear(self.m_hidden_size, self.m_cont_vocab_size)

        self.m_bow2contvocab = nn.Linear(self.m_latent_size, self.m_cont_vocab_size)
        self.m_output2contvocab = nn.Linear(self.m_hidden_size, self.m_cont_vocab_size)

        self.m_l_output = nn.Linear(self.m_l_hidden_size, self.m_word_type_num)

        if self.m_de_strategy == "attn":
            self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

        # self = self.to(self.m_device)
        
    def f_init_user_item_word(self, E_network):
        E_network.eval()

        self.m_user_embedding.weight.requires_grad=False
        self.m_item_embedding.weight.requires_grad=False

        self.m_user_embedding.weight.data.copy_(E_network.m_user_embedding.weight.data)
        self.m_item_embedding.weight.data.copy_(E_network.m_item_embedding.weight.data)

    def f_encode_z(self, input_bow, user_item):
        input_x_bow_embedding = self.m_bow_input_layer(input_bow)

        mu_z = self.m_mu_fc(input_x_bow_embedding+user_item)
        logvar_z = self.m_logvar_fc(input_x_bow_embedding+user_item)
        std_z = torch.exp(0.5*logvar_z)

        z = std_z*torch.randn_like(std_z)+mu_z

        return z, mu_z, logvar_z
    
    def f_decode_z(self, z):
        bow_logits = self.m_bow2contvocab(z)

        return bow_logits

    def f_reparameterize_prior(self, user_item):
        mu_p_z = self.m_mu_p_fc(user_item)
        logvar_p_z = self.m_logvar_p_fc(user_item)

        std_p_z = torch.exp(0.5*logvar_p_z)

        return mu_p_z, logvar_p_z

    def forward(self, input_sequence, input_bow, input_l_sequence, user_ids, item_ids, random_flag):

        ### input_l_sequence: batch_size*seq_len

        batch_size = input_sequence.size(0)
        de_batch_size = input_sequence.size(0)
        de_len = input_sequence.size(1)

        input_x_embedding = self.m_embedding(input_sequence)

        if torch.isinf(input_x_embedding).any():
            print(" ... input_x_embedding inf ")
        
        if torch.isnan(input_x_embedding).any():
            print(" ... input_x_embedding nan ")

        input_l_embedding = self.m_l_embedding(input_l_sequence)

        input_user_hidden = self.m_user_embedding(user_ids)
        input_item_hidden = self.m_item_embedding(item_ids)

        if torch.isinf(input_user_hidden).any():
            print(" ... input_user_hidden inf ")

        if torch.isnan(input_user_hidden).any():
            print(" ... input_user_hidden nan ")

        if torch.isinf(input_item_hidden).any():
            print(" ... input_item_hidden inf ")

        if torch.isnan(input_item_hidden).any():
            print(" ... input_item_hidden nan ")

        user_item_hidden_init = input_user_hidden+input_item_hidden
        
        output_logits = []
        output_l_logits = []

        user_item_hidden_i = self.m_latent2embed(user_item_hidden_init)

        ### z: batch_size*latent_size
        z, mu_z, logvar_z = self.f_encode_z(input_bow, user_item_hidden_i)

        bow_logits = self.f_decode_z(z)

        mu_p_z, logvar_p_z = self.f_reparameterize_prior(user_item_hidden_i)

        hidden = None
        decode_strategy = self.m_de_strategy

        hidden_l = None
        
        if decode_strategy == "attn":
            """
            attention mechanism output
            """

            for de_step_i in range(de_len):
                input_x_step_i = input_x_embedding[:, de_step_i, :]
                
                input_l_step_i = input_l_embedding[:, de_step_i, :]
            
                output_l_step_i, hidden_l = self.m_decoder_l_rnn(torch.cat([input_l_step_i, input_x_step_i], dim=-1).unsqueeze(1), hidden_l)

                output_l_step_i = output_l_step_i.squeeze(1)
                output_l_logit_i = self.m_l_output(output_l_step_i)
                output_l_logits.append(output_l_logit_i.unsqueeze(1))

                input_x_step_i = input_x_step_i+input_l_sequence[:, de_step_i].unsqueeze(1)*user_item_hidden_i

                output_step_i, hidden = self.m_decoder_rnn(input_x_step_i.unsqueeze(1), hidden)

                output_step_i = output_step_i.squeeze(1)

                output_func_logit_i = self.m_output2funcvocab(output_step_i)
                output_func_logit_i[:, self.m_func_vocab_size:] = float('-inf')

                output_logit_i = (1-input_l_sequence[:, de_step_i].unsqueeze(1))*output_func_logit_i
                
                output_logit_i += input_l_sequence[:, de_step_i].unsqueeze(1)*self.m_output2contvocab(output_step_i)

                output_logit_bow_i = self.m_bow2contvocab(z)

                output_logit_i = output_logit_i+input_l_sequence[:, de_step_i].unsqueeze(1)*output_logit_bow_i

                output_logits.append(output_logit_i.unsqueeze(1))
                
                user_attn = torch.cat([output_step_i, input_user_hidden], dim=-1)
                item_attn = torch.cat([output_step_i, input_item_hidden], dim=-1)
            
                user_attn_score = self.m_attn(user_attn)
                item_attn_score = self.m_attn(item_attn)

                attn_score = F.softmax(torch.cat([user_attn_score, item_attn_score], dim=-1), dim=-1)

                user_item_hidden_i = attn_score[:, 0].unsqueeze(1)*input_user_hidden
                user_item_hidden_i = user_item_hidden_i+attn_score[:, 1].unsqueeze(1)*input_item_hidden
                user_item_hidden_i = self.m_latent2embed(user_item_hidden_i)

            logits = torch.cat(output_logits, dim=1)
            l_logits = torch.cat(output_l_logits, dim=1)

        ### logits: batch_size*seq_len*voc_size
        logits = logits.contiguous()
        l_logits = l_logits.contiguous()

        logits = logits.view(-1, logits.size(-1))
        l_logits = l_logits.view(-1, l_logits.size(-1))

        return logits, l_logits, bow_logits, mu_z, logvar_z, mu_p_z, logvar_p_z

