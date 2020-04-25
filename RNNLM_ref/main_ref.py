import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from collections import Counter
import os
from argparse import Namespace
import datetime

flags = Namespace(
    train_file = "clothing.txt",
    seq_size = 32,
    batch_size = 16,
    embedding_size = 256, 
    lstm_size = 300,
    gradients_norm = 5, 
    initial_words = ['i', "am"],
    predict_top_k = 5,
    checkpoint_path = 'checkpoint',
    model_name = 'RNNLM_ref'
)

def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r') as f:
        text = f.read()

    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k:w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w:k for k, w in int_to_vocab.items()}

    n_vocab = len(int_to_vocab)

    print("vocab", n_vocab)
    
    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text)/(seq_size*batch_size))
    print('batch', num_batches)
    print("int_text", len(int_text))

    in_text = int_text[:num_batches*batch_size*seq_size]
    out_text = np.zeros_like(in_text)
    print("out_text", out_text.shape)

    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]

    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size*batch_size)
    length = [seq_size for i in range(batch_size)]
    for i in range(0, num_batches*seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size], length

import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class REVIEWDI(nn.Module):

    def __init__(self, vocab_size, seq_size, embedding_size, hidden_size):
        super().__init__()

        # self.m_device=device

        self.m_embedding_size = embedding_size

        self.m_hidden_size = hidden_size
        
        self.m_vocab_size = vocab_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        # self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, batch_first=True)

        self.m_linear_output = nn.Linear(self.m_hidden_size, self.m_vocab_size)
    
        # self = self.to(self.m_device)

    def forward(self, input_sequence, hidden, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        input_embedding = self.m_embedding(input_sequence)

        hidden = hidden.unsqueeze(0)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        outputs, _ = self.m_decoder_rnn(packed_input, hidden)

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        # print("padded_outputs", padded_outputs.size())

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        output_logit = padded_outputs.view(-1, padded_outputs.size(2))

        # print("output_logit", output_logit.size())
        output_logit = self.m_linear_output(output_logit)

        return output_logit

    def zero_state(self, batch_size):
        hidden = torch.randn([batch_size, self.m_hidden_size])
        return hidden

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)

    net = REVIEWDI(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)

    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.001)

    iteration = 0

    file_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
    tensor_file_name = file_time
    print("tensor_file_name", tensor_file_name)
    tensor_writer = SummaryWriter("../tensorboard/"+flags.model_name+"/"+tensor_file_name)

    epoch_num = 50
    for e in range(epoch_num):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        # state_h, state_c = net.zero_state(flags.batch_size)

        # state_h = state_h.to(device)
        # state_c = state_c.to(device)

        for x, y, length in batches:
            
            # state_h, state_c = net.zero_state(flags.batch_size)

            # state_h = state_h.to(device)
            # state_c = state_c.to(device)
            state_h = net.zero_state(flags.batch_size)
            state_h = state_h.to(device)

            iteration += 1

            net.train()

            optimizer.zero_grad()

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            length = torch.tensor(length).to(device)

            logits = net(x, state_h, length)

            y = y[:, :torch.max(length).item()].contiguous().view(-1)
            # logits = logits.view(-1, pred.size(1))

            # NLL_loss = self.m_XE(pred, target)
            loss = criterion(logits.view(-1, logits.size(1)), y)

            # state_h = state_h.detach()
            # state_c = state_c.detach()
            
            loss_value = loss.item()

            loss.backward()

            # _ = torch.nn.utils.clip_grad_norm_(net.parameters(), flags.gradients_norm)

            optimizer.step()

            # print(iteration)
            scalar_name = "loss"
            scalar = loss_value
            index = iteration
            tensor_writer.add_scalar('./data/'+scalar_name, scalar, index)

            if iteration % 100 == 0:
                print('epoch: {}/{}'.format(e, epoch_num), 'Iteration:{}'.format(iteration), 'loss:{}'.format(loss_value))

            # if iteration % 1000 == 0:
                # predict(device, net, n_vocab, vocab_to_int, int_to_vocab, top_k=5)
                # torch.save(net.state_dict(), 'checkpoint_pt/model-best.pt')

    tensor_writer.close()

def predict(device, net, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    # state_h, state_c = net.zero_state(1)
    state_h = net.zero_state(1)
    state_h = state_h.to(device)
    # state_c = state_c.to(device)

    words = ['i', "am"]
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, state_h )

    _, top_ix = torch.topk(output[0], k=top_k)

    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])
    
    print("+"*10, len(words), "+"*10)
    print(' '.join(words))
    print("*"*10, len(words), "*"*10)
    
if __name__ == "__main__":
    main()




