import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, vocab, args):
        super().__init__()

        self.m_lambda_kl = args.lambda_kl
        self.m_vocab = vocab
        self.m_embedding_size = args.dim_emb

        self.m_hidden_size = args.dim_h
        # self.m_word_dropout_rate = args.word_dropout
        self.m_embedding_dropout = args.dropout
        self.m_latent_size = args.dim_z

        self.m_num_layers = args.nlayers
        # self.m_rnn_type = args.rnn_type

        self.m_vocab_size = self.m_vocab.size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_encoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=True, batch_first=False)
        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, num_layers=self.m_num_layers, bidirectional=False, batch_first=False)

        self.m_hidden2mean_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)
        self.m_hidden2logv_z = nn.Linear(self.m_hidden_size*2, self.m_latent_size)

        self.m_latent2hidden = nn.Linear(self.m_latent_size, self.m_embedding_size)

        self.m_output2vocab = nn.Linear(self.m_hidden_size, self.m_vocab_size)

        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def forward(self, input, is_train=False):
        # print("input size", input.size())
        # input = input.transpose(0, 1)
        input_emb = self.m_embedding_dropout(self.m_embedding(input))
        
        _, h = self.m_encoder_rnn(input_emb)

        # print("h", h.size())
        h = torch.cat([h[-2], h[-1]], 1)
        # print("h", h.size())

        mu = self.m_hidden2mean_z(h)
        logvar = self.m_hidden2logv_z(h)

        # print("mu", mu.size())
        # exit()

        z_std = torch.exp(0.5*logvar)
        pre_z = torch.randn_like(z_std)
        z = pre_z*z_std + mu

        input_embedding = self.m_embedding_dropout(self.m_embedding(input))

        init_de_hidden = self.m_latent2hidden(z)
        # print("z", z.size())
        # print("init_de_hidden", init_de_hidden.size())
        # print("input_embedding", input_embedding.size())

        # exit()
        repeat_hidden_0 = init_de_hidden.unsqueeze(0)
        repeat_hidden_0 = repeat_hidden_0.expand(input_embedding.size(0), init_de_hidden.size(0), init_de_hidden.size(-1))

        input_embedding = input_embedding + repeat_hidden_0
        # input_embedding = input_embedding+init_de_hidden

        hidden = None
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)

        output = output.contiguous()
        # print("output size", output.size())
        logits = self.m_output2vocab(output.view(-1, output.size(-1)))

        logits = logits.view(output.size(0), output.size(1), -1)

        return mu, logvar, z, logits

    def loss(self, logits, targets, mu, logvar):
        rec_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.m_vocab.pad, reduction='none').view(targets.size())

        rec_loss = rec_loss.sum(dim=0).mean()

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

        loss = rec_loss+self.m_lambda_kl*kl_loss

        losses = {}
        losses['rec'] = rec_loss 
        losses['kl'] = kl_loss
        losses['loss'] = loss

        return losses

    def step(self, losses):
        self.opt.zero_grad()
        # print(losses)
        losses['loss'].backward()

        self.opt.step()

    def decode(self, z, input, hidden=None):
        # input = input.transpose(0, 1)
        input_embedding = self.m_embedding_dropout(self.m_embedding(input))

        init_de_hidden = self.m_latent2hidden(z)

        repeat_hidden_0 = init_de_hidden.unsqueeze(0)
        repeat_hidden_0 = repeat_hidden_0.expand(input_embedding.size(0), init_de_hidden.size(0), init_de_hidden.size(-1))

        input_embedding = input_embedding+repeat_hidden_0

        hidden = None
        output, hidden = self.m_decoder_rnn(input_embedding, hidden)

        output = output.contiguous()
        # print("output size", output.size())
        logits = self.m_output2vocab(output.view(-1, output.size(-1)))

        logits = logits.view(output.size(0), output.size(1), -1)

        return logits, hidden

    def flatten(self):
        self.m_encoder_rnn.flatten_parameters()
        self.m_decoder_rnn.flatten_parameters()

    def generate(self, z, max_len, alg):
        sents = []

        # input = torch.zeros(len(z), 1, device=z.device).fill_(self.m_vocab.go).long()
        input = torch.zeros(1, len(z), device=z.device).fill_(self.m_vocab.go).long()

        hidden = None

        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            # print("logits size", logits.size())
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
                # print("input,", input.size())
            
        return torch.cat(sents)


    