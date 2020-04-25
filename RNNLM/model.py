import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class REVIEWDI(nn.Module):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device=device

        self.m_embedding_size = args.embedding_size

        self.m_hidden_size = args.hidden_size
        
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_embedding = nn.Embedding(self.m_vocab_size, self.m_embedding_size)
        # self.m_embedding_dropout = nn.Dropout(p=self.m_embedding_dropout)

        self.m_decoder_rnn = nn.GRU(self.m_embedding_size, self.m_hidden_size, batch_first=True)

        self.m_linear_output = nn.Linear(self.m_hidden_size, self.m_vocab_size)
    
        self = self.to(self.m_device)

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)

        input_sequence = input_sequence[sorted_idx]

        hidden = torch.randn([batch_size, self.m_hidden_size]).to(self.m_device)

        input_embedding = self.m_embedding(input_sequence)

        hidden = hidden.unsqueeze(0)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        
        outputs, _ = self.m_decoder_rnn(packed_input, hidden)

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        output_logit = padded_outputs.view(-1, padded_outputs.size(2))

        output_logit = self.m_linear_output(output_logit)

        return output_logit


        

            
