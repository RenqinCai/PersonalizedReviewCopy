import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class _NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device
        print("self.m_device", self.m_device)
        
        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_unk_idx = vocab_obj.unk_idx

        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size

        self.m_embedding_size = args.embedding_size
        self.m_layers_num = args.layers_num
        self.m_latent_size = args.latent_size
        self.m_hidden_size = args.hidden_size
        self.m_dropout_rate = args.dropout

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_dropout_rate)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_latent_size)
        self.m_item_embedding = nn.Embedding(self.m_item_size, self.m_latent_size)

        self.m_user_item_encoder = _ENC_NETWORK(self.m_embedding, self.m_output2vocab, vocab_obj, args, self.m_device)

        print("self.m_device", self.m_device)
        self.m_user_num = torch.zeros((self.m_user_size, 1)).to(self.m_device)
        self.m_item_num = torch.zeros((self.m_item_size, 1)).to(self.m_device)

        self.m_generator = _GEN_NETWORK(self.m_embedding, self.m_output2vocab, self.m_user_embedding, self.m_item_embedding, vocab_obj, args, self.m_device)

        self = self.to(self.m_device)

    def forward(self, reviews, review_lens, user_ids, item_ids, random_flag, forward_flag):

        if forward_flag == "encode":
            return self.encode(reviews, review_lens, user_ids, item_ids)

        elif forward_flag == "decode":
            return self.decode(reviews, user_ids, item_ids, random_flag)

        return None
        ### map user & item hidden to user embedding and item embedding 
        
        ### use user & item embedding as input to the generator
        
    def encode(self, reviews, review_lens, user_ids, item_ids):
        user_logits, item_logits = self.m_user_item_encoder(reviews, review_lens, user_ids, item_ids)

        return user_logits, item_logits

    def update_user_item(self, reviews, review_lens, user_ids, item_ids):
        user_hidden = self.m_user_item_encoder.m_user_encoder(reviews, review_lens)
        item_hidden = self.m_user_item_encoder.m_item_encoder(reviews, review_lens)

        for i, user_id in enumerate(user_ids):
            user_id = user_id.item()
            self.m_user_embedding.weight.data[user_id] += user_hidden[i].detach()
            self.m_user_num[user_id] += 1.0

        for i, item_id in enumerate(item_ids):
            item_id = item_id.item()
            self.m_item_embedding.weight.data[item_id] += item_hidden[i].detach()
            self.m_item_num[item_id] += 1.0

        # self.m_user_embedding.weight.data[user_ids] += user_hidden
        # self.m_item_embedding.weight.data[item_ids] += item_hidden

    def normalize_user_item(self):
        self.m_user_embedding.weight.data = self.m_user_embedding.weight.data/self.m_user_num
        self.m_item_embedding.weight.data = self.m_item_embedding.weight.data/self.m_item_num

    def decode(self, reviews, user_ids, item_ids, random_flag):
        logits = self.m_generator(reviews, user_ids, item_ids, random_flag)

        return logits
    
class _GEN_NETWORK(nn.Module):
    def __init__(self, embedding, output2vocab, user_embedding, item_embedding, vocab_obj, args, device):
        super().__init__()

        self.m_device = device

        self.m_hidden_size = args.hidden_size
        self.m_latent_size = args.latent_size

        self.m_max_sequence_len = args.max_seq_length
        self.m_num_layers = args.num_layers
        self.m_bidirectional = args.bidirectional
        self.m_embedding_size = args.embedding_size

        self.m_sos_idx = vocab_obj.sos_idx
        self.m_eos_idx = vocab_obj.eos_idx
        self.m_pad_idx = vocab_obj.pad_idx
        self.m_unk_idx = vocab_obj.unk_idx
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_embedding = embedding 
        self.m_user_embedding = user_embedding
        self.m_item_embedding = item_embedding

        self.m_tanh = nn.Tanh()
        self.m_latent2output = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_de_strategy = args.de_strategy

        self.m_src_mask = None
        self.m_pos_encoder = PositionalEncoding(self.m_embedding_size)
        encoder_layers = TransformerEncoderLayer(self.m_embedding_size, self.m_head_num, self.m_hidden_size)
        self.m_transformer_decoder = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        # self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)
        
        # self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)
        self.m_output2vocab = output2vocab

        if self.m_de_strategy == "gate":
            self.m_decoder_gate = nn.Sequential(nn.Linear(self.m_hidden_size, 1), nn.Sigmoid())

        if self.m_de_strategy == "attn":
            self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

    def f_generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def f_init_weights(self):
        initrange = 0.1
        self.m_transformer_decoder.bias.data.zero_()
        self.m_transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_de_sequence, user_ids, item_ids, random_flag):
        
        # if not hasattr(self, '_flattened'):
        #     self.m_decoder_rnn.flatten_parameters()
        #     setattr(self, '_flattened', True)

        de_batch_size = input_de_sequence.size(0)
        de_len = input_de_sequence.size(1)

        input_de_embedding = self.m_embedding(input_de_sequence)

        input_user_hidden = self.m_user_embedding(user_ids)
        input_item_hidden = self.m_item_embedding(item_ids)

        user_item_hidden = self.m_latent2output(input_user_hidden+input_item_hidden)
        input_de_embedding = user_item_hidden+input_de_embedding

        ### seq_len*batch_size*embedding_size
        input_de_embedding = input_de_embedding.transpose(0, 1)

        hidden = None
        decode_strategy = self.m_de_strategy

        if self.m_src_mask is None or self.m_src_mask.size(0) != de_len:
            mask = self.f_generate_square_subsequent_mask(de_len).to(self.m_device)
            self.m_src_mask = mask
        
        input_de_embedding = input_de_embedding*math.sqrt(self.m_embedding_size)
        input_de_embedding = self.m_pos_encoder(input_de_embedding)

        output = self.m_transformer_decoder(input_de_embedding, self.m_src_mask)
        output = output.transpose(0, 1)

        ### output: batch_size*seq_len*hidden_size
        output = output.contiguous()
        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.m_dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float()*(-math.log(10000.0)/embedding_size))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('m_pe', pe)

    def forward(self, x):
        x = x+self.m_pe[:x.size(0), :]
        return self.m_dropout(x)

### obtain the representation of users and items. 
class _ENC_NETWORK(nn.Module):
    def __init__(self, embedding, output2vocab, vocab_obj, args, device):
        super(_ENC_NETWORK, self).__init__()

        self.m_device = device
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_hidden_size = args.hidden_size
        self.m_layers_num = args.layers_num
        self.m_dropout_rate = args.dropout
        self.m_latent_size = args.latent_size

        self.m_embedding_size = args.embedding_size

        self.m_user_encoder = _ENCODER(embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_encoder = _ENCODER(embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_user_decoder = _DECODER(embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_decoder = _DECODER(embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_output2vocab = output2vocab
        # self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size) 

    def forward(self, reviews, review_lens, user_ids, item_ids):

        ### obtain user representation
        user_hidden = self.m_user_encoder(reviews, review_lens)

        ### obtain item representation
        item_hidden = self.m_item_encoder(reviews, review_lens)

        # self.m_user_embedding.weight.data[user_ids] += user_hidden
        # self.m_item_embedding.weight.data[item_ids] += item_hidden

        user_output = self.m_user_decoder(reviews, user_hidden)
        item_output = self.m_item_decoder(reviews, item_hidden)

        user_logits = self.m_output2vocab(user_output.view(-1, user_output.size(2)))
        item_logits = self.m_output2vocab(item_output.view(-1, item_output.size(2)))

        return user_logits, item_logits

class _ENCODER(nn.Module):
    def __init__(self, embedding, latent_size, hidden_size, device, layers_num=1, dropout=0.3):
        super(_ENCODER, self).__init__()
        
        self.m_dropout_rate = dropout
        self.m_device = device

        self.m_embedding = embedding
        self.m_embedding_dropout = nn.Dropout(self.m_dropout_rate)

        self.m_latent_size = latent_size
        self.m_hidden_size = hidden_size
        self.m_layers_num = layers_num

        self.m_gru = nn.GRU(self.m_hidden_size, self.m_hidden_size, self.m_layers_num, dropout=self.m_dropout_rate, bidirectional=True)
        
        self.m_hidden2latent = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        
    def forward(self, x, x_len, hidden=None):
        # if not hasattr(self, '_flattened'):
        #     self.m_gru.flatten_parameters()
        #     setattr(self, '_flattened', True)

        batch_size = x.size(0)
        input_embedding = self.m_embedding(x)
        input_embedding = self.m_embedding_dropout(input_embedding)

        encoder_outputs, _ = self.m_gru(input_embedding)

        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (x_len-1).long()

        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        en_latent = self.m_hidden2latent(last_en_hidden)

        return en_latent

class _DECODER(nn.Module):
    def __init__(self, embedding, embedding_size, latent_size, hidden_size, device, layers_num=1, dropout=0.3):
        super(_DECODER, self).__init__()
        self.m_device = device

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
