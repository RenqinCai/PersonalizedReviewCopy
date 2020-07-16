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

        self.m_latent_size = args.latent_size

        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_size
        self.m_item_num = vocab_obj.item_size

        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_latent_size)

        self.m_decoder = None

    def init_tokenizer_decoder(self, tokenizer_decoder, decoder):
        self.m_eos_idx = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.m_pad_idx = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]

        self.m_decoder = decoder

    def f_init_user_item_word(self, E_network):
            E_network.eval()

            self.m_user_embedding.weight.requires_grad=False
            self.m_item_embedding.weight.requires_grad=False

            self.m_user_embedding.weight.data.copy_(E_network.m_user_embedding.weight.data)
            self.m_item_embedding.weight.data.copy_(E_network.m_item_embedding.weight.data)
        
    def forward(self, inputs):

        reconstruction_mask = (inputs!=50257).float()
        sent_length = torch.sum(reconstruction_mask, dim=1)

        batch_size = inputs.size(0)

        latent_z = torch.zeros((batch_size, self.m_latent_size)).normal_().to(inputs.device)

        outputs = self.m_decoder(input_ids=inputs, past=latent_z, labels=inputs, label_ignore=self.m_pad_idx)

        loss_rec = outputs[0]
        
        loss = loss_rec.mean()

        return loss

# class _GEN_NETWORK(nn.Module):
#     def __init__(self, vocab_obj, args):
#         super().__init__()

#         # self.m_device = device

#         self.m_hidden_size = args.hidden_size
#         self.m_latent_size = args.latent_size
        
#         self.m_max_sequence_len = args.max_seq_length
#         self.m_layers_num = args.layers_num
#         self.m_bidirectional = args.bidirectional
#         self.m_embedding_size = args.embedding_size
#         # self.m_aspect_num = args.aspect_num

#         self.m_sos_idx = vocab_obj.sos_idx
#         self.m_eos_idx = vocab_obj.eos_idx
#         self.m_pad_idx = vocab_obj.pad_idx
#         self.m_unk_idx = vocab_obj.unk_idx
#         self.m_vocab_size = vocab_obj.vocab_size
#         self.m_user_num = vocab_obj.user_size
#         self.m_item_num = vocab_obj.item_size

#         self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
#         self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_latent_size)
#         self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_latent_size)

#         # self.m_aspect_embedding = nn.Embedding(self.m_aspect_size, self.m_latent_size)

#         self.m_tanh = nn.Tanh()
#         self.m_latent2output = nn.Linear(self.m_latent_size, self.m_embedding_size)

#         self.m_de_strategy = args.de_strategy

#         self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_layers_num, bidirectional=False, batch_first=True)
        
#         self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

#         if self.m_de_strategy == "attn":
#             self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

#         # self = self.to(self.m_device)
        
#     def f_init_user_item_word(self, E_network):
#         E_network.eval()

#         # if torch.isinf(E_network.module.m_user_embedding.weight).any():
#         #     for i, debug in enumerate(E_network.module.m_user_embedding.weight):
#         #         if torch.isinf(debug).any():
#         #             print(i, "E network user_embedding debug", debug)
#         #             exit()
#             # print("self.m_user_embedding", self.m_user_embedding.weight)
        
#         self.m_embedding.weight.requires_grad=False
#         self.m_user_embedding.weight.requires_grad=False
#         self.m_item_embedding.weight.requires_grad=False
#         self.m_output2vocab.weight.requires_grad=False

#         self.m_embedding.weight.data.copy_(E_network.m_embedding.weight.data)

#         self.m_user_embedding.weight.data.copy_(E_network.m_user_embedding.weight.data)
        
#         self.m_item_embedding.weight.data.copy_(E_network.m_item_embedding.weight.data)
       
#         self.m_output2vocab.weight.data.copy_(E_network.m_output2vocab.weight.data)
        
#         if torch.isinf(self.m_embedding.weight).any():
#             for i, debug in enumerate(self.m_embedding.weight):
#                 if torch.isinf(debug).any():
#                     print(i, "self.m_embedding debug", debug)
#                     exit()
#             print("input_embedding", self.m_embedding.weight)

#         if torch.isinf(self.m_user_embedding.weight).any():
#             for i, debug in enumerate(self.m_user_embedding.weight):
#                 if torch.isinf(debug).any():
#                     print(i, "self.m_user_embedding debug", debug)
#                     exit()
#             print("self.m_user_embedding", self.m_user_embedding.weight)

#         if torch.isinf(self.m_item_embedding.weight).any():
#             for i, debug in enumerate(self.m_item_embedding.weight):
#                 if torch.isinf(debug).any():
#                     print(i, "self.m_item_embedding debug", debug)
#                     exit()
#             print("self.m_item_embedding", self.m_item_embedding.weight)

#         if torch.isinf(self.m_output2vocab.weight).any():
#             for i, debug in enumerate(self.m_output2vocab.weight):
#                 if torch.isinf(debug).any():
#                     print(i, "self.m_output2vocab debug", debug)
#                     exit()
#             print("self.m_output2vocab", self.m_output2vocab.weight)

#     def forward(self, input_de_sequence, user_ids, item_ids, random_flag):

#         batch_size = input_de_sequence.size(0)
#         de_batch_size = input_de_sequence.size(0)
#         de_len = input_de_sequence.size(1)

#         input_de_embedding = self.m_embedding(input_de_sequence)

#         if torch.isnan(input_de_embedding).any():
#             print("input_embedding", input_de_embedding)

#         input_user_hidden = self.m_user_embedding(user_ids)
#         input_item_hidden = self.m_item_embedding(item_ids)

#         if torch.isnan(input_user_hidden).any():
#             print("self.m_user_embedding nan", input_user_hidden)

#         if torch.isinf(input_user_hidden).any():
#             for i, debug in enumerate(input_user_hidden):
#                 if torch.isinf(debug).any():
#                     print(i, "m_user_embedding debug", debug)
#                     exit()
#             print("self.m_user_embedding inf", input_user_hidden)

#         if torch.isnan(input_item_hidden).any():
#             print("self.m_item_embedding nan", input_item_hidden)
        
#         if torch.isinf(input_item_hidden).any():
#             print("input_item_hidden inf", input_item_hidden)

#         user_item_hidden_init = input_user_hidden+input_item_hidden

#         if torch.isnan(user_item_hidden_init).any():
#             print("... user_item_hidden_i nan ...", user_item_hidden_init)
#             exit()
        
#         if torch.isinf(user_item_hidden_init).any():
#             print("+++ user_item_hidden_i inf 1 +++", user_item_hidden_init)
#             exit()

#         if (user_item_hidden_init != user_item_hidden_init).any():
#             print("... user_item_hidden_i inf ...", user_item_hidden_init)
#             exit()
        
#         output = []
#         user_item_hidden_i = self.m_latent2output(user_item_hidden_init)

#         if torch.isnan(self.m_latent2output.weight).any():
#             print("self.m_latent2output.weight", self.m_latent2output.weight)
#             exit()
        
#         if torch.isnan(user_item_hidden_i).any():
#             for i, debug in enumerate(user_item_hidden_i):
#                 if torch.isnan(debug).any():
#                     print("weight", self.m_latent2output.weight.data[i])
#                     print("user_item_hidden_init", user_item_hidden_init[i])
#                     print(i, " user_item_hidden_i debug", debug)
#                     exit()

#         hidden = None
#         decode_strategy = self.m_de_strategy
        
#         if decode_strategy == "attn":
#             """
#             attention mechanism output
#             """

#             for de_step_i in range(de_len):
#                 input_de_step_i = input_de_embedding[:, de_step_i, :]
#                 input_de_step_i = input_de_step_i + user_item_hidden_i
#                 input_de_step_i = input_de_step_i.unsqueeze(1)
                
#                 if torch.isnan(input_de_step_i).any():
#                     print("input_de_step_i", input_de_step_i)
#                     exit()

#                 output_step_i, hidden = self.m_decoder_rnn(input_de_step_i, hidden)
#                 output.append(output_step_i)

#                 if torch.isnan(hidden).any():
#                     print("hidden", hidden)
#                     exit()

#                 if torch.isnan(output_step_i).any():
#                     for _, output_row_step_i in enumerate(output_step_i):
#                         if torch.isnan(output_row_step_i).any():
#                             print("output_row_step_i", output_row_step_i)
#                             exit()

#                 output_step_i = output_step_i.squeeze(1)
                
#                 user_attn = torch.cat([output_step_i, input_user_hidden], dim=-1)
#                 item_attn = torch.cat([output_step_i, input_item_hidden], dim=-1)
            
#                 user_attn_score = self.m_attn(user_attn)
#                 item_attn_score = self.m_attn(item_attn)

#                 attn_score = F.softmax(torch.cat([user_attn_score, item_attn_score], dim=-1), dim=-1)

#                 user_item_hidden_i = attn_score[:, 0].unsqueeze(1)*input_user_hidden
#                 user_item_hidden_i = user_item_hidden_i+attn_score[:, 1].unsqueeze(1)*input_item_hidden
#                 user_item_hidden_i = self.m_latent2output(user_item_hidden_i)

#             output = torch.cat(output, dim=1)

#         output = output.contiguous()
#         logits = self.m_output2vocab(output.view(-1, output.size(2)))
        
#         if torch.isnan(logits).any():
#             print("logits", logits)
#             exit()

#         return logits

### obtain the representation of users and items. 

