import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

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

        self.m_cluster_num = args.cluster_num

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_dropout_rate)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size, bias=False)

        # self.m_user_embedding = nn.Embedding(self.m_user_size, self.m_latent_size)

        user_embedding = torch.zeros(self.m_latent_size, self.m_cluster_num).to(self.m_device)
        cluster_size = torch.zeros(self.m_cluster_num).to(self.m_device)
        avg_user_embedding = user_embedding.clone()
        user_cluster_prob = torch.zeros(self.m_user_size, self.m_cluster_num).to(self.m_device)

        self.register_buffer("m_user_cluster_prob", user_cluster_prob)
        self.register_buffer("m_user_embedding", user_embedding)
        self.register_buffer("m_cluster_size", cluster_size)
        self.register_buffer("m_avg_user_embedding", avg_user_embedding)

        self.m_item_embedding = nn.Embedding(self.m_item_size, self.m_latent_size)

        self.m_user_item_encoder = _ENC_NETWORK(self.m_embedding, self.m_user_embedding, self.m_cluster_size, self.m_avg_user_embedding, self.m_output2vocab, vocab_obj, args, self.m_device)

        print("self.m_device", self.m_device)
        # self.m_user_num = torch.zeros((self.m_user_size, 1)).to(self.m_device)
        
        self.m_item_cnt = torch.zeros((self.m_item_size, 1)).to(self.m_device)

        self.m_generator = _GEN_NETWORK(self.m_embedding, self.m_output2vocab, self.m_user_embedding, self.m_user_cluster_prob, self.m_item_embedding, vocab_obj, args, self.m_device)

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
        user_logits, item_logits, user_quantize_diff = self.m_user_item_encoder(reviews, review_lens, user_ids, item_ids)

        return user_logits, item_logits, user_quantize_diff

    def update_user_item(self, reviews, review_lens, user_ids, item_ids):
        user_hidden = self.m_user_item_encoder.m_user_encoder(reviews, review_lens)

        user_hidden_sum = user_hidden.pow(2).sum(1, keepdim=True)
        user_embedding_sum = self.m_user_embedding.pow(2).sum(0, keepdim=True)

        dist = user_hidden_sum-2*user_hidden@self.m_user_embedding+user_embedding_sum

        _, cluster_ids = (-dist).max(1)

        user_one_hot = F.one_hot(user_ids, self.m_user_size).type(user_hidden.dtype)
        cluster_one_hot = F.one_hot(cluster_ids, self.m_cluster_num).type(user_hidden.dtype)

        user_cluster_cnt = user_one_hot.transpose(0, 1) @ cluster_one_hot
        self.m_user_cluster_prob.add_(user_cluster_cnt)

        item_hidden = self.m_user_item_encoder.m_item_encoder(reviews, review_lens)
        item_one_hot = F.one_hot(item_ids, self.m_item_size).type(item_hidden.dtype)
        item_embedding_sum = item_one_hot.transpose(0, 1) @ item_hidden

        self.m_item_embedding.weight.data.add_(item_embedding_sum)

        self.m_item_cnt.add_(torch.sum(item_one_hot, dim=0).unsqueeze(1))

    def normalize_user_item(self):
        # self.m_user_embedding.weight.data = self.m_user_embedding.weight.data/self.m_user_num

        # print("user_cnt", self.m_user_cluster_prob[0])
        user_cnt = torch.sum(self.m_user_cluster_prob, dim=1, keepdim=True)
        # print("user_cnt", user_cnt[0])
        self.m_user_cluster_prob /= user_cnt

        self.m_item_embedding.weight.data = self.m_item_embedding.weight.data/self.m_item_cnt

        self.m_item_embedding.weight.requires_grad=False
        self.m_embedding.weight.requires_grad = False
        self.m_output2vocab.weight.requires_grad = False

    def decode(self, reviews, user_ids, item_ids, random_flag):
        logits = self.m_generator(reviews, user_ids, item_ids, random_flag)

        return logits
    
class _GEN_NETWORK(nn.Module):
    def __init__(self, embedding, output2vocab, user_embedding, user_cluster_prob, item_embedding, vocab_obj, args, device):
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
        self.m_user_cluster_prob = user_cluster_prob
        self.m_item_embedding = item_embedding

        self.m_tanh = nn.Tanh()
        self.m_latent2output = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_de_strategy = args.de_strategy

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)
        
        # self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)
        self.m_output2vocab = output2vocab

        if self.m_de_strategy == "gate":
            self.m_decoder_gate = nn.Sequential(nn.Linear(self.m_hidden_size, 1), nn.Sigmoid())

        if self.m_de_strategy == "attn":
            self.m_attn= nn.Linear(self.m_hidden_size, self.m_latent_size)
            # self.m_attn = nn.Sequential(nn.Linear(self.m_hidden_size+self.m_latent_size, self.m_hidden_size), nn.Tanh(), nn.Linear(self.m_hidden_size, 1)) 

        # self = self.to(self.m_device)

    def forward(self, input_de_sequence, user_ids, item_ids, random_flag):
        
        batch_size = input_de_sequence.size(0)
        de_batch_size = input_de_sequence.size(0)
        de_len = input_de_sequence.size(1)

        input_de_embedding = self.m_embedding(input_de_sequence)

        ### batch_size*cluster_size
        input_user_cluster_prob = self.m_user_cluster_prob[user_ids]
        
        weighted_user_hidden = input_user_cluster_prob @ self.m_user_embedding.transpose(0, 1)

        input_user_hidden = weighted_user_hidden
        # input_user_hidden = self.m_user_embedding(user_ids)
        input_item_hidden = self.m_item_embedding(item_ids)

        user_item_hidden_init = input_user_hidden+input_item_hidden
        
        output = []
        user_item_hidden_i = self.m_latent2output(user_item_hidden_init)

        hidden = None
        decode_strategy = self.m_de_strategy
        
        for de_step_i in range(de_len):
            input_de_step_i = input_de_embedding[:, de_step_i, :]
            input_de_step_i = input_de_step_i + user_item_hidden_i
            input_de_step_i = input_de_step_i.unsqueeze(1)
            output_step_i, hidden = self.m_decoder_rnn(input_de_step_i, hidden)
            output.append(output_step_i)

            ### batch_size*hidden_size
            output_step_i = output_step_i.squeeze(1)
            
            user_attn_score = self.m_attn(output_step_i) @ self.m_user_embedding
            # user_attn = torch.cat([output_step_i, input_user_hidden], dim=-1)
            # item_attn = torch.cat([output_step_i, input_item_hidden], dim=-1)
        
            # user_attn_score = self.m_attn(user_attn)
            # item_attn_score = self.m_attn(item_attn)

            ### user_attn_score: batch_size*cluster_num
            # user_attn_score = output_step_i @ self.m_user_embedding
        
            item_attn = self.m_attn(output_step_i) * input_item_hidden
            item_attn_score = torch.sum(item_attn, dim=-1, keepdim=True)
            
            ### user_attn_score: 
            ### user_cluster_prob: batch_size*cluster_num 
            user_attn_score = user_attn_score*input_user_cluster_prob

            # print("item_attn_score", item_attn_score.size())
            # print("user_attn_score", user_attn_score.size())

            ### attn_score: batch_size*(cluster_num+1)
            attn_score = F.softmax(torch.cat([user_attn_score, item_attn_score], dim=-1), dim=-1)

            ### user_item_hidden_i: batch_size*cluster_num
            user_item_hidden_i = (attn_score[:, :-1]*input_user_cluster_prob) @ (self.m_user_embedding.transpose(0, 1))

            user_item_hidden_i = user_item_hidden_i+attn_score[:, -1].unsqueeze(1)*input_item_hidden

            user_item_hidden_i = self.m_latent2output(user_item_hidden_i)

        output = torch.cat(output, dim=1)
        output = output.contiguous()
        logits = self.m_output2vocab(output.view(-1, output.size(2)))
        
        return logits

### obtain the representation of users and items. 
class _ENC_NETWORK(nn.Module):
    def __init__(self, embedding, user_embedding, cluster_size, avg_user_embedding, output2vocab, vocab_obj, args, device):
        super(_ENC_NETWORK, self).__init__()

        self.m_device = device
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_hidden_size = args.hidden_size
        self.m_layers_num = args.layers_num
        self.m_dropout_rate = args.dropout
        self.m_latent_size = args.latent_size
        self.m_cluster_num = args.cluster_num
        self.m_decay = args.decay
        # self.m_eps = args.eps
        self.m_eps = 1e-10

        self.m_embedding_size = args.embedding_size
        self.m_commitment = args.commitment

        self.m_user_embedding = user_embedding
        self.m_cluster_size = cluster_size
        self.m_avg_user_embedding = avg_user_embedding

        self.m_user_encoder = _ENCODER(embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_encoder = _ENCODER(embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_user_decoder = _DECODER(embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_decoder = _DECODER(embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_output2vocab = output2vocab
        
    def forward(self, reviews, review_lens, user_ids, item_ids):

        ### user_hidden: batch_size*hidden_size
        user_hidden = self.m_user_encoder(reviews, review_lens)

        ### user_hidden_sum: batch_size*1
        user_hidden_sum = user_hidden.pow(2).sum(1, keepdim=True)
        
        ### user_embedding_sum: 1*cluster_size
        user_embedding_sum = self.m_user_embedding.pow(2).sum(0, keepdim=True)

        ### m_user_embedding: latent_size*cluster_size

        ### cross_term: batch_size*cluster_size
        cross_term = -2*user_hidden@self.m_user_embedding
        
        ### batch_size*cluster_size
        dist = user_hidden_sum+cross_term+user_embedding_sum

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.m_cluster_num).type(user_hidden.dtype)
        user_quantize = self.embed_code(embed_ind)

        print("--"*20)
        print("user_embedding_sum", user_embedding_sum)
        # print("cross_term", cross_term)
        # print("embed_ind", embed_ind)
        print("--"*20)

        if self.training:
            # print("---"*20, "training")
            self.m_cluster_size.data.mul_(self.m_decay).add_(1-self.m_decay, embed_onehot.sum(0))
            embed_sum = user_hidden.transpose(0, 1) @ embed_onehot
            self.m_avg_user_embedding.data.mul_(self.m_decay).add_(1-self.m_decay, embed_sum)
            n = self.m_cluster_size.sum()
            cluster_size = ((self.m_cluster_size+self.m_eps)/(n+self.m_cluster_num*self.m_eps)*n)

            embed_normalized = self.m_avg_user_embedding/cluster_size.unsqueeze(0)
            self.m_user_embedding.data.copy_(embed_normalized)

        user_quantize_diff = self.m_commitment*torch.mean((user_hidden - user_quantize.detach()).pow(2))
        ### obtain item representation
        item_hidden = self.m_item_encoder(reviews, review_lens)

        user_output = self.m_user_decoder(reviews, user_quantize)
        item_output = self.m_item_decoder(reviews, item_hidden)

        user_logits = self.m_output2vocab(user_output.view(-1, user_output.size(2)))
        item_logits = self.m_output2vocab(item_output.view(-1, item_output.size(2)))

        return user_logits, item_logits, user_quantize_diff

    def embed_code(self, embed_ind):
        return F.embedding(embed_ind, self.m_user_embedding.transpose(0, 1))

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

class _USER_ITEM_ENCODER(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(_USER_ITEM_ENCODER, self).__init__()

        self.m_device = device
        self.m_user_size = vocab_obj.user_size
        self.m_item_size = vocab_obj.item_size
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_hidden_size = args.hidden_size
        self.m_layers_num = args.layers_num
        self.m_dropout_rate = args.dropout
        self.m_latent_size = args.latent_size
        self.m_cluster_num = args.cluster_num
        self.m_decay = args.decay
        # self.m_eps = args.eps
        self.m_eps = 1e-10

        self.m_embedding_size = args.embedding_size
        self.m_commitment = args.commitment

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_dropout_rate)

        self.m_user_encoder = _ENCODER(self.m_embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_encoder = _ENCODER(self.m_embedding, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_user_decoder = _DECODER(self.m_embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)
        self.m_item_decoder = _DECODER(self.m_embedding, self.m_embedding_size, self.m_latent_size, self.m_hidden_size, self.m_device, self.m_layers_num, self.m_dropout_rate)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        self = self.to(self.m_device)

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
