import enum
import torch
from torch import log
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

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

        self.m_output_hidden_size = args.output_hidden_size

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_user_linear = nn.Linear(self.m_user_embed_size, self.m_output_hidden_size)
        self.m_item_linear = nn.Linear(self.m_item_embed_size, self.m_output_hidden_size)
        self.m_attr_linear = nn.Linear(self.m_attr_embed_size, self.m_output_hidden_size)

        self.m_lambda = 0.28
        self.m_user_output = nn.Linear(1, 1)
        self.m_attr_item_linear = nn.Linear(1, 1)
        self.m_attr_user_linear = nn.Linear(1, 1)

        self = self.to(self.m_device)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids):
        # print("==="*10)

        """ item """

        attr_item_x = self.m_attr_embedding(attr_item)
        attr_item_x = attr_item_x.transpose(0, 1)
        # print("attr_item_x", attr_item_x.size())
        # print(attr_item_x)

        attr_item_mask = self.f_generate_mask(attr_lens_item)
        # print("attr_item_mask", attr_item_mask.size())
        # print(attr_item_mask)

        attr_attn_item = self.m_attn(attr_item_x, src_key_padding_mask = attr_item_mask)
        attr_attn_item = attr_attn_item.transpose(0, 1)

        attr_item = self.m_attr_linear(attr_attn_item)

        user_x = self.m_user_embedding(user_ids)
        user_x = self.m_user_linear(user_x)

        user_x = user_x.unsqueeze(2)
        
        user_attr_item = torch.matmul(attr_item, user_x).squeeze()
        # print("user_attr_item", user_attr_item.size())
        # user_attr_item_logits = user_attr_item.unsqueeze(2)

        attr_item_weight = attr_tf_item.unsqueeze(2)
        attr_item_logits = self.m_attr_item_linear(attr_item_weight)

        # user_attr_item_logits = user_attr_item_logits.squeeze(-1)
        attr_item_logits = attr_item_logits.squeeze(-1)
        user_attr_item_logits = user_attr_item + attr_item_logits

        user_attr_item_logits = user_attr_item_logits.unsqueeze(-1)

        weighted_user_attr_item_logits = user_attr_item_logits*attr_item
        
        # print("**"*15)
        if torch.isnan(weighted_user_attr_item_logits).any():
            print("user_attr_item_output", weighted_user_attr_item_logits)

        weighted_attr_item_mask = attr_item_mask.unsqueeze(-1).expand(weighted_user_attr_item_logits.size())
        weighted_attr_item_mask = weighted_attr_item_mask

        weighted_user_attr_item_logits.data.masked_fill_(weighted_attr_item_mask.data, -float('inf'))
        user_attr_item_output = torch.max(weighted_user_attr_item_logits, 1)[0]

        if torch.isnan(user_attr_item_output).any():
            print("user_attr_item_output", user_attr_item_output)

        if torch.isinf(user_attr_item_output).any():
            print("user_attr_item_output inf", user_attr_item_output)

        if torch.isnan(weighted_user_attr_item_logits).any():
            print("user_attr_item_output", weighted_user_attr_item_logits)

        # """ user """

        attr_user_x = self.m_attr_embedding(attr_user) 
        attr_user_x = attr_user_x.transpose(0, 1)

        attr_user_mask = self.f_generate_mask(attr_lens_user)
    
        attr_attn_user = self.m_attn(attr_user_x, src_key_padding_mask=attr_user_mask)
        attr_attn_user = attr_attn_user.transpose(0, 1)

        attr_user = self.m_attr_linear(attr_attn_user)

        item_x = self.m_item_embedding(item_ids)
        item_x = self.m_item_linear(item_x)
        item_x = item_x.unsqueeze(2)

        item_attr_user = torch.matmul(attr_user, item_x).squeeze()

        # item_attr_user_logits = item_attr_user.unsqueeze(2)

        attr_user_weight = attr_tf_user.unsqueeze(2)
        attr_user_logits = self.m_attr_user_linear(attr_user_weight)

        # item_attr_user_logits = item_attr_user_logits.squeeze(-1)
        attr_user_logits = attr_user_logits.squeeze(-1)
        item_attr_user_logits = item_attr_user + attr_user_logits

        item_attr_user_logits = item_attr_user_logits.unsqueeze(-1)
        
        weighted_item_attr_user_logits = item_attr_user_logits*attr_user

        weighted_attr_user_mask = attr_user_mask.unsqueeze(-1).expand(weighted_item_attr_user_logits.size())
        weighted_attr_user_mask = weighted_attr_user_mask

        weighted_item_attr_user_logits.data.masked_fill_(weighted_attr_user_mask.data, -float('inf'))
        item_attr_user_output = torch.max(weighted_item_attr_user_logits, 1)[0]

        output = user_attr_item_output + item_attr_user_output

        if torch.isnan(item_attr_user_output).any():
            print("item_attr_user_output", item_attr_user_output)

        if torch.isnan(output).any():
            print("output", output)

        return output
    
    def f_pred_forward(self, user_item_output, pos_targets, pos_lens, neg_targets, neg_lens):
        ### neg_targets: batch_size*neg_num
        ### neg_embed: batch_size*neg_num*embed_size
        neg_embed = self.m_attr_embedding(neg_targets)

        ### user_item_output: batch_size*hidden_size
        ### neg_logits: batch_size*neg_num
        neg_logits = torch.matmul(neg_embed, user_item_output.unsqueeze(-1))
        neg_logits = neg_logits.squeeze(-1)

        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        pos_embed = self.m_attr_embedding(pos_targets)
        # print("pos_embed", pos_embed.size())
        # print("user_item_output", user_item_output.size())

        ### user_item_output: batch_size*hidden_size*1
        ### pos_logits: batch_size*pos_num
        pos_logits = torch.matmul(pos_embed, user_item_output.unsqueeze(-1))
        pos_logits = pos_logits.squeeze(-1)

        if torch.isnan(pos_logits).any():
            print("pos_logits", pos_logits)

        # print("pos len size", pos_lens.size())
        # print("neg len size", neg_lens.size())

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        # print("pos_logits", pos_logits.size())
        # print("neg_logits", neg_logits.size())

        # print("pos_mask", pos_mask.size())
        # print("neg_mask", neg_mask.size())

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    # def f_eval_forward(self, user_item_output, pos_targets, pos_lens, neg_targets, neg_lens):
    #     ### neg_targets: batch_size*neg_num
    #     ### neg_embed: batch_size*neg_num*embed_size
    #     neg_embed = self.m_attr_embedding(neg_targets)

    #     ### user_item_output: batch_size*hidden_size
    #     ### neg_logits: batch_size*neg_num
    #     neg_logits = torch.matmul(neg_embed, user_item_output.unsqueeze(-1))
    #     neg_logits = neg_logits.squeeze(-1)

    #     neg_mask = self.f_generate_mask(neg_lens)
    #     neg_mask = ~neg_mask

    #     ### targets: batch_size*pos_num
    #     ### pos_embed: batch_size*pos_num*embed_size
    #     pos_embed = self.m_attr_embedding(pos_targets)
    #     # print("pos_embed", pos_embed.size())
    #     # print("user_item_output", user_item_output.size())

    #     ### user_item_output: batch_size*hidden_size*1
    #     ### pos_logits: batch_size*pos_num
    #     pos_logits = torch.matmul(pos_embed, user_item_output.unsqueeze(-1))
    #     pos_logits = pos_logits.squeeze(-1)

    #     if torch.isnan(pos_logits).any():
    #         print("pos_logits", pos_logits)

    #     # print("pos len size", pos_lens.size())
    #     # print("neg len size", neg_lens.size())

    #     pos_mask = self.f_generate_mask(pos_lens)
    #     pos_mask = ~pos_mask

    #     # print("pos_logits", pos_logits.size())
    #     # print("neg_logits", neg_logits.size())

    #     # print("pos_mask", pos_mask.size())
    #     # print("neg_mask", neg_mask.size())

    #     logits = torch.cat([pos_logits, neg_logits], dim=-1)

    #     mask = torch.cat([pos_mask, neg_mask], dim=-1)

    #     new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

    #     new_targets = new_targets*mask

    #     return logits, mask, new_targets


        