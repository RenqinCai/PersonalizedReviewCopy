"""
attention decoder and u, i encoder.
alpha*U+(1-alpha)*V
"""
import operator
from numpy.lib.arraysetops import isin
import torch
from torch import log, unsqueeze
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np
import time
from queue import PriorityQueue

class RNN_DECODER(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(RNN_DECODER, self).__init__()
        self.m_device = device

        self.m_hidden_size = args.output_hidden_size
        self.m_attr_embed_size = args.attr_emb_size

        self.m_vocab_size = vocab_obj.vocab_size
        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_rnn = nn.GRUCell(self.m_attr_embed_size, self.m_attr_embed_size)

        self.m_attention = LUONG_GATE_ATTN(self.m_attr_embed_size, self.m_device)

        self.m_output_attr_embedding = nn.Linear(self.m_attr_embed_size, self.m_vocab_size)

        self.m_dropout = nn.Dropout(p=0.2)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        torch.nn.init.uniform_(self.m_output_attr_embedding.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)

    def forward(self, input, hidden):

        # print("input", input.size())
        # print("=="*10)
        attr_embed = self.m_attr_embedding(input)

        # print("input", input)
        # print("attr_embed", attr_embed.size())
        # print("state", hidden.size())

        hidden = self.m_rnn(attr_embed, hidden)

        # print("hidden", hidden)

        # print("hidden", hidden.size())
        ### output: batch_size*embed_size
        output, attn_weights = self.m_attention(hidden)

        # print("output", output.size())

        ### logit: batch_size*voc_size
        logit = self.m_output_attr_embedding(output)

        return logit, hidden, attn_weights

class LUONG_GATE_ATTN(nn.Module):
    def __init__(self, hidden_size, device, prob=0.1):
        super(LUONG_GATE_ATTN, self).__init__()

        self.m_linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))

        self.m_linear_output = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), nn.SELU(), nn.Dropout(p=prob))

        self.m_softmax = nn.Softmax(dim=-1)
        self.m_dropout = nn.Dropout(p=prob)

        self.f_init_weight()

        self.m_device = device
        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        for name, module in self.m_linear_in.named_modules():
            if(isinstance(module, nn.Linear)):
                torch.nn.init.uniform_(module.weight, -initrange, initrange)
                if module.bias is not None:
                    torch.nn.init.uniform_(module.bias, -initrange, initrange)

        for name, module in self.m_linear_output.named_modules():
            if(isinstance(module, nn.Linear)):
                torch.nn.init.uniform_(module.weight, -initrange, initrange)
                if module.bias is not None:
                    torch.nn.init.uniform_(module.bias, -initrange, initrange)

    def init_context(self, context):
        self.m_context = context

    def forward(self, h):
        ### h: batch_size*hidden_size
        ### gamma_h: batch_size*embed_size
        gamma_h = self.m_linear_in(h)

        ### m_context: batch_size*2*embed_size
        ### weights: batch_size*2
        weights = self.m_dropout(torch.matmul(self.m_context, gamma_h.unsqueeze(-1))).squeeze(-1)
        weights = self.m_softmax(weights)
        # print("weights", weights.size())

        ### weights: batch_size*2
        ### c_t: batch_size*2*embed_size
        c_t = weights.unsqueeze(-1)*self.m_context

        # print("self.m_context", self.m_context.size())

        # print("c_t 1", c_t.size())
        ### c_t: batch_size*embed_size
        c_t = torch.sum(c_t, dim=1)

        # print("c_t 2", c_t.size())

        # print("h", h)
        # print("c", c_t)

        ### [h, c_t]: batch_size*(2*embed_size)
        ### output: batch_size*embed_size
        output = self.m_linear_output(torch.cat([h, c_t], dim=1))

        # print("output", output.size())

        return output, weights

class _ATTR_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(_ATTR_NETWORK, self).__init__()

        self.m_device = device
        
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num

        self.m_attr_embed_size = args.attr_emb_size
        self.m_user_embed_size = args.user_emb_size
        self.m_item_embed_size = args.item_emb_size

        self.m_attn_head_num = args.attn_head_num
        self.m_attn_layer_num = args.attn_layer_num

        self.m_attn_linear_size = args.attn_linear_size

        self.m_sos = 1
        self.m_eos = 2

        ### encoder
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        ### decoder
        self.m_decoder = RNN_DECODER(vocab_obj, args, device)

        self.m_logsoftmax = nn.LogSoftmax(dim=-1)
        
        # self.m_attr_embedding_x = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1

        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, user_ids, item_ids, input_targets):

        ### encode
        ### user_embed: batch_size*embed_size
        batch_size = input_targets.size(0)
        seq_len = input_targets.size(1)

        user_embed = self.m_user_embedding(user_ids) 

        ### item_embed: batch_size*embed_size                             
        item_embed = self.m_item_embedding(item_ids)

        ### context: batch_size*2*embed_size
        context = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)

        ### target_inputs: batch_size*seq_len
        # ### targets: batch_size*seq_len
        
        logits = []    
        logit = None
        
        self.m_decoder.m_attention.init_context(context)

        ### state: batch_size*2*embed_size
        # state = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)
        ### state: batch_size*embed_size
        init_hidden = user_embed+item_embed

        ### input_targets: batch_size*seq_len
        # seq_len = input_targets.size(1)

        for i in range(seq_len):
            ### input_target_i: batch_size
            input_target_i = input_targets[:, i]

            ### logit: batch_size*voc_size
            logit, init_hidden, _ = self.m_decoder(input_target_i, init_hidden)   
            logits.append(logit.unsqueeze(1))

            # print("logit", logit.size())

        ### logits: batch_size*seq_len*voc_size
        logits = torch.cat(logits, dim=1)

        for i in range(batch_size):
            for j in range(seq_len):
                logits[i, j, input_targets[i, :j+1]] = -1e7

        # for i in range(batch_size):
        #     input_target_i = input_targets[i]
        #     logits[i, :, input_targets[i]] = -1e7

        # print(logits.size())
        
        return logits

    def f_eval_forward(self, user_ids, item_ids):
        ### encode
        ### user_embed: batch_size*embed_size
        user_embed = self.m_user_embedding(user_ids) 

        ### item_embed: batch_size*embed_size                             
        item_embed = self.m_item_embedding(item_ids)

        ### context: batch_size*2*(embed_size)
        context = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)

        ### hidden: batch_size*2*(embed_size)
        # hidden = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1)], dim=1)

        ### init_hidden: batch_size*embed_size
        init_hidden = user_embed+item_embed

        generations = self.f_beam_decode(init_hidden, context)

        return generations

    def f_beam_decode(self, decoder_hidden, context):
        beam_width = 5

        topk = 1

        decoded_batch = []

        batch_size = context.size(0)

        max_step_num = 1

        ### context: batch_size*2*embed_size

        ### decoder_hidden: batch_size*2*hidden_size

        for i in range(batch_size):
            # print("==="*10, i, "==="*10)
            ### decoder_hidden_i: 1*hidden_size
            decoder_hidden_i = decoder_hidden[i].unsqueeze(0)

            ### context[i]: 2*embed_size
            ### context_i: 2*1*embed_size

            context_i = context[i].unsqueeze(0)
            # print("context_i", context_i.size())

            self.m_decoder.m_attention.init_context(context_i)

            decoder_input_i = torch.zeros(1).fill_(self.m_sos).to(self.m_device).long()

            endnodes = []
            end_number_required = 1
            done_flag = False

            node = BeamSearchNode(decoder_hidden_i, None, decoder_input_i, 0, 1)
            nodes_queue = PriorityQueue()

            # score = -node.f_eval()
            nodes_queue.put((-node.f_eval(), node))
            qsize = 1

            next_beam_width = 1

            for step_i in range(max_step_num):
                if qsize > 2000: break
                # print("next_beam_width", next_beam_width)
                nextnodes = []
                for beam_i in range(next_beam_width):
                    score, node = nodes_queue.get()

                    decoder_input_i = node.m_wordid
                    decoder_hidden_i = node.m_h

                    if node.m_wordid.item() == self.m_eos and node.m_prev_node != None:
                        endnodes.append((score, node))

                        if len(endnodes) >= end_number_required:
                            done_flag = True
                            break
                        else:
                            continue
                    
                    pre_i = []
                    pre_i.append(decoder_input_i.item())
                    pre_node = node.m_prev_node
                    while pre_node != None:
                        pre_i.append(pre_node.m_wordid.item())
                        pre_node = pre_node.m_prev_node

                    pre_i = torch.tensor(pre_i).to(decoder_hidden.device)

                    ### decoder_logits_i: batch_size*voc_size
                    decoder_logits_i, decoder_hidden_i, weight_i = self.m_decoder(decoder_input_i, decoder_hidden_i)

                    decoder_logits_i[:, pre_i] = -1e7

                    ### decoder_probs_i: batch_size*voc_size
                    decoder_probs_i = torch.log_softmax(decoder_logits_i, dim=-1)

                    topk_logprob, topk_indices = torch.topk(decoder_probs_i, beam_width)
                
                    for new_k in range(beam_width):
                        decode_k = topk_indices[0][new_k].view(1)
                        log_p_k = topk_logprob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden_i, node, decode_k, node.m_logp+log_p_k, node.m_leng+1)
                        score = -node.f_eval()

                        nextnodes.append((score, node))

                if done_flag:
                    break

                while not nodes_queue.empty():
                    score, node = nodes_queue.get()

                next_beam_width = beam_width-len(endnodes)

                for next_i in range(len(nextnodes)):
                    score, nn = nextnodes[next_i]
                    nodes_queue.put((score, nn))

                qsize += len(nextnodes) - 1

            if len(endnodes) == 0:
                endnodes = [nodes_queue.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                # utterance.append(n.m_wordid.item())

                while n.m_prev_node != None:
                    utterance.append(n.m_wordid.item())
                    n = n.m_prev_node

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch

class BeamSearchNode(object):
    def __init__(self, hidden_state, previous_node, wordid, logprob, length):
        self.m_h = hidden_state
        self.m_prev_node = previous_node

        self.m_wordid = wordid
        self.m_logp = logprob
        self.m_leng = length
    
    def f_eval(self, alpha=1.0):
        reward = 0

        epsilon = np.random.uniform(1e-6, 1e-20)

        return self.m_logp/float(self.m_leng-1+epsilon)+alpha*reward