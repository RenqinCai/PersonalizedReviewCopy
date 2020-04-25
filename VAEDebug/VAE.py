import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class VAE(nn.Module):
    def __init__(self, vocab, args, device):
        super().__init__()

        self.m_device = device
        self.m_lambda_kl = args.lambda_kl
        self.m_vocab = vocab
        self.m_embedding_size = args.dim_emb

        self.m_hidden_size = args.dim_h
        # self.m_word_dropout_rate = args.word_dropout
        self.m_embedding_dropout_rate = args.dropout
        self.m_latent_size = args.dim_z

        self.m_num_layers = args.nlayers
        # self.m_rnn_type = args.rnn_type

        self.m_vocab_size = self.m_vocab.size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout_rate)

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=True)

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size*2, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=True)

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        # self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.opt = optim.Adam(self.parameters(), lr=args.lr)

        self.m_NLL = nn.NLLLoss(size_average=False, ignore_index=self.m_vocab.pad).to(self.m_device)

        self = self.to(self.m_device)
        
    def encode(self, input, length):
        input = input.transpose(0, 1)
        # input = self.m_embedding_dropout(self.m_embedding(input))

        batch_size = input.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input[sorted_idx]

        input_embedding = self.m_embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        encoder_outputs, hidden = self.m_encoder_rnn(packed_input)

        encoder_outputs = rnn_utils.pad_packed_sequence(encoder_outputs, batch_first=True)[0]
        encoder_outputs = encoder_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        encoder_outputs = encoder_outputs[reversed_idx]

        # first_dim_index = torch.arange(batch_size).to(.m_device)
        first_dim_index = torch.arange(batch_size).to(self.m_device)
        second_dim_index = (length-1).long()

        last_en_hidden = encoder_outputs[first_dim_index, second_dim_index, :].contiguous()

        return self.m_hidden2mean_z(last_en_hidden), self.m_hidden2logv_z(last_en_hidden)
    
    def decode(self, z, input, length, hidden=None):
        input = input.transpose(0, 1)
        input_embedding = self.m_embedding_dropout(self.m_embedding(input))

        # print("input", input)
        init_de_hidden = self.m_latent2hidden(z)
        # print("init_de_hidden", init_de_hidden)

        repeat_hidden_0 = init_de_hidden.unsqueeze(1)
        repeat_hidden_0 = repeat_hidden_0.expand(init_de_hidden.size(0), input_embedding.size(1), init_de_hidden.size(-1))

        # print("0 input_embedding", input_embedding)
        # print("repeat_hidden_0", repeat_hidden_0)
        # input_embedding = input_embedding+repeat_hidden_0
        input_embedding = torch.cat([input_embedding, repeat_hidden_0], dim=-1)

        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)

        # input_embedding = input_embedding[sorted_idx]

        # # print("sorted_lengths", sorted_lengths)
        # # # print("sorted_idx", sorted_idx)

        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        # # # # # outputs, _ = self.m_decoder_rnn(packed_input, init_de_hidden)
        # padded_outputs, hidden = self.m_decoder_rnn(packed_input, hidden)

        # padded_outputs = rnn_utils.pad_packed_sequence(padded_outputs, batch_first=True)[0]
        # padded_outputs = padded_outputs.contiguous()

        # _, reversed_idx = torch.sort(sorted_idx)
        # # # print("reversed_idx", reversed_idx)
        # # # print(sorted_idx[reversed_idx])
        # output = padded_outputs[reversed_idx]

        # exit()

        # init_de_hidden = self.m_latent2hidden(z)

        # repeat_hidden_0 = init_de_hidden.unsqueeze(1)
        # repeat_hidden_0 = repeat_hidden_0.expand(init_de_hidden.size(0), input_embedding.size(1), init_de_hidden.size(-1))

        # input_embedding = input_embedding + repeat_hidden_0

        # # hidden = None

        # print("input_embedding", input_embedding)
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)
        output = output.contiguous()

        # print("output", output)

        logits = self.m_output2vocab(output.view(-1, output.size(-1)))
        # print("logits", logits)

        logits = logits.view(output.size(0), output.size(1), -1)

        return logits, hidden

    def forward(self, input, length, targets, is_train=False):

        mu, logvar = self.encode(input, length)
        # print("mu", mu)
        # z = reparameterize(mu, logvar)
        z_std = torch.exp(0.5*logvar)
        pre_z = torch.randn_like(z_std)
        # print("pre_z", pre_z)
        
        z = pre_z*z_std + mu
        # print("z", z)
        
        logits, _ = self.decode(z, input, length)

        # rec_loss = self.loss_rec(logits, targets).mean()
        rec_loss = self.loss_rec(logits, targets) / len(mu)
        # print("rec_loss", rec_loss)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)
        # kl_loss = oss_kl(mu, logvar)
        # print("kl loss", kl_loss)
        
        # exit()
        return {'rec': rec_loss, 'kl':kl_loss}
        
        # print("input size", input.size())
        # input = input.transpose(0, 1)
        # input_emb = self.m_embedding_dropout(self.m_embedding(input))
        
        # _, h = self.m_encoder_rnn(input_emb)

        # # print("h", h.size())
        # h = torch.cat([h[-2], h[-1]], 1)
        # # print("h", h.size())

        # mu = self.m_hidden2mean_z(h)
        # logvar = self.m_hidden2logv_z(h)

        # print("mu", mu.size())
        # exit()

        # input_embedding = self.m_embedding_dropout(self.m_embedding(input))

        # init_de_hidden = self.m_latent2hidden(z)
        # print("z", z.size())
        # print("init_de_hidden", init_de_hidden.size())
        # print("input_embedding", input_embedding.size())

        # exit()
        # repeat_hidden_0 = init_de_hidden
        # repeat_hidden_0 = init_de_hidden.unsqueeze(0)
        # repeat_hidden_0 = repeat_hidden_0.expand(input_embedding.size(0), init_de_hidden.size(0), init_de_hidden.size(-1))

        # input_embedding = input_embedding + repeat_hidden_0
        # input_embedding = input_embedding+init_de_hidden

        # hidden = None
        # output, hidden = self.m_decoder_rnn(input_embedding, hidden)

        # output = output.contiguous()
        # print("output size", output.size())
        # logits = self.m_output2vocab(output.view(-1, output.size(-1)))

        # logits = logits.view(output.size(0), output.size(1), -1)

        # return mu, logvar, z, logits

    # def loss(self, logits, targets, mu, logvar):
    #     rec_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.m_vocab.pad, reduction='none').view(targets.size())

    #     rec_loss = rec_loss.sum(dim=0).mean()

    #     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    #     loss = rec_loss+self.m_lambda_kl*kl_loss

    #     losses = {}
    #     losses['rec'] = rec_loss 
    #     losses['kl'] = kl_loss
    #     losses['loss'] = loss

    #     return losses

    def loss_rec(self, logits, targets):
        targets = targets.transpose(0, 1)
        targets = targets.contiguous()

        # print("logit",logits.size())
        # print("target", targets.size())
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.m_vocab.pad, reduction='none')
        # loss = loss.view(targets.size())

        pred = nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)

        # logp = logp.view(batch_size, -1, self.m_vocab_size)
        targets = targets.view(-1)
        # print("target", targets)
        # print("pred", pred)
        # pred = pred.view(-1, pred.size(2))

        loss = self.m_NLL(pred, targets)
        # print("loss", loss)
        # loss = 

        # # loss = loss.sum(dim=0)
        # loss = loss.sum(dim=1)

        return loss

    def loss(self, losses):
        return losses['rec']+self.m_lambda_kl*losses['kl']

    def step(self, losses):
        self.opt.zero_grad()
        # print(losses)
        losses['loss'].backward()

        self.opt.step()

    def flatten(self):
        self.m_encoder_rnn.flatten_parameters()
        self.m_decoder_rnn.flatten_parameters()

    def generate(self, z, max_len, alg):
        sents = []

        # input = torch.zeros(len(z), 1, device=z.device).fill_(self.m_vocab.go).long()
        input = torch.zeros(1, len(z), device=z.device).fill_(self.m_vocab.go).long()

        hidden = None

        length = torch.ones(len(z), device=z.device).long()
        # print(length)
        for l in range(max_len):
            
            sents.append(input)
            logits, hidden = self.decode(z, input, length, hidden)
            # print("logits size", logits.size())
            if alg == 'greedy':
                input = logits.argmax(dim=-1).transpose(0, 1)
                # print("input,", input.size())
                # exit()
        # exit()
        return torch.cat(sents)


    