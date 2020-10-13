import enum
import torch
from torch import log, unsqueeze
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

        self.m_attn_linear_size = args.attn_linear_size

        self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        self.m_exp_user = nn.Linear(1, 1)
        self.m_log_user = nn.Linear(1, 1)
        self.m_linear_user = nn.Linear(1, 1)
        
        self.m_exp_item = nn.Linear(1, 1)
        self.m_log_item = nn.Linear(1, 1)
        self.m_linear_item = nn.Linear(1, 1)

        self.m_attr_user = nn.Linear(self.m_attr_embed_size, 4)
        self.m_attr_item = nn.Linear(self.m_attr_embed_size, 4)

        self.m_bias_tf = nn.Linear(1, 1, bias=False)

        self.f_init_weight()

        self = self.to(self.m_device)

    # def f_init_weight(self):
    #     torch.nn.init.normal_(self.m_output_attr_embedding.weight, 0.0, 0.1)
    #     torch.nn.init.normal_(self.m_attr_embedding.weight, 0.0, 0.1)
    #     # torch.nn.init.normal_(self.m_tag_item_embedding.weight, 0.0, 0.01)
    #     torch.nn.init.normal_(self.m_user_embedding.weight, 0.0, 0.1)
    #     torch.nn.init.normal_(self.m_item_embedding.weight, 0.0, 0.1)

    def f_init_weight(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.m_output_attr_embedding_user.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_output_attr_embedding_item.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_attr_embedding.weight, -initrange, initrange)
        # torch.nn.init.normal_(self.m_tag_item_embedding.weight, 0.0, 0.01)
        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids, pos_targets, pos_lens, neg_targets, neg_lens):
        # print("==="*10)

        """ item """

        attr_item_embed = self.m_attr_embedding(attr_item)
        attr_item_embed = attr_item_embed.transpose(0, 1)
        # print("attr_item_x", attr_item_x.size())
        # print(attr_item_x)

        attr_item_mask = self.f_generate_mask(attr_lens_item)
        # print("attr_item_mask", attr_item_mask.size())
        # print(attr_item_mask)

        attr_attn_item = self.m_attn(attr_item_embed, src_key_padding_mask = attr_item_mask)
        attr_attn_item = attr_attn_item.transpose(0, 1)

        ### attr_attn_item: batch_size*attr_len_item*(head_num*attr_embed)
        ### attr_item: batch_size*attr_len_item*output_size
        # attr_item = self.m_attr_linear(attr_attn_item)

        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)

        weighted_user_attr_item_logits = attr_attn_item
        
        # weighted_attr_item_mask = attr_item_mask.unsqueeze(-1).expand(weighted_user_attr_item_logits.size())
        # weighted_user_attr_item_logits.data.masked_fill_(weighted_attr_item_mask.data, -float('inf'))
        # user_attr_item_output = torch.max(weighted_user_attr_item_logits, 1)[0]

        weighted_attr_item_mask = attr_item_mask
        user_attr_item_output = weighted_user_attr_item_logits.sum(1)/((~weighted_attr_item_mask).sum(1).unsqueeze(-1)
        )
        
        # """ user """

        attr_user_embed = self.m_attr_embedding(attr_user) 
        attr_user_embed = attr_user_embed.transpose(0, 1)

        attr_user_mask = self.f_generate_mask(attr_lens_user)
    
        attr_attn_user = self.m_attn(attr_user_embed, src_key_padding_mask=attr_user_mask)
        attr_attn_user = attr_attn_user.transpose(0, 1)

        item_embed = self.m_item_embedding(item_ids)

        weighted_item_attr_user_logits = attr_attn_user

        # weighted_attr_user_mask = attr_user_mask.unsqueeze(-1).expand(weighted_item_attr_user_logits.size())
        # weighted_item_attr_user_logits.data.masked_fill_(weighted_attr_user_mask.data, -float('inf'))
        # item_attr_user_output = torch.max(weighted_item_attr_user_logits, 1)[0]

        weighted_attr_user_mask = attr_user_mask
        item_attr_user_output = weighted_item_attr_user_logits.sum(1)/((~weighted_attr_user_mask).sum(1).unsqueeze(-1))

        # user_output = user_embed
        # item_output = item_embed

        user_output = item_attr_user_output
        item_output = user_attr_item_output

        # user_output = (1.0-self.m_gamma)*user_embed+self.m_gamma*item_attr_user_output
        # item_output = (1.0-self.m_gamma)*item_embed+self.m_gamma*user_attr_item_output

        neg_embed_user = self.m_output_attr_embedding_user(neg_targets)
        neg_embed_item = self.m_output_attr_embedding_item(neg_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        neg_logits_user = torch.matmul(neg_embed_user, user_output.unsqueeze(-1))
        neg_logits_user = neg_logits_user.squeeze(-1)

        neg_logits_item = torch.matmul(neg_embed_item, item_output.unsqueeze(-1))
        neg_logits_item = neg_logits_item.squeeze(-1)

        # print("neg_lens", neg_lens)
        # exit()
        neg_mask = self.f_generate_mask(neg_lens)
        neg_mask = ~neg_mask

        ### targets: batch_size*pos_num
        ### pos_embed: batch_size*pos_num*embed_size
        # pos_embed = self.m_attr_embedding(pos_targets)

        pos_embed_user = self.m_output_attr_embedding_user(pos_targets)
        pos_embed_item = self.m_output_attr_embedding_item(pos_targets)

        ### user_item_output: batch_size*ouput_size
        ### neg_logits: batch_size*neg_num
        pos_logits_user = torch.matmul(pos_embed_user, user_output.unsqueeze(-1))
        pos_logits_user = pos_logits_user.squeeze(-1)

        pos_logits_item = torch.matmul(pos_embed_item, item_output.unsqueeze(-1))
        pos_logits_item = pos_logits_item.squeeze(-1)

        pos_logits = pos_logits_user+pos_logits_item
        neg_logits = neg_logits_user+neg_logits_item

        pos_mask = self.f_generate_mask(pos_lens)
        pos_mask = ~pos_mask

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        mask = torch.cat([pos_mask, neg_mask], dim=-1)

        new_targets = torch.cat([torch.ones_like(pos_targets), torch.zeros_like(neg_targets)], dim=1)

        new_targets = new_targets*mask

        return logits, mask, new_targets

    def f_eval_forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids):

        """ item """

        attr_item_embed = self.m_attr_embedding(attr_item)
        attr_item_embed = attr_item_embed.transpose(0, 1)
        # print("attr_item_x", attr_item_x.size())
        # print(attr_item_x)

        attr_item_mask = self.f_generate_mask(attr_lens_item)

        attr_attn_item = self.m_attn(attr_item_embed, src_key_padding_mask = attr_item_mask)
        attr_attn_item = attr_attn_item.transpose(0, 1)

        ### attr_attn_item: batch_size*attr_len_item*(head_num*attr_embed)
        ### attr_item: batch_size*attr_len_item*output_size
        # attr_item = self.m_attr_linear(attr_attn_item)

        ### user_x: batch_size*user_embed
        user_embed = self.m_user_embedding(user_ids)

        weighted_user_attr_item_logits = attr_attn_item
        
        # weighted_attr_item_mask = attr_item_mask.unsqueeze(-1).expand(weighted_user_attr_item_logits.size())
        # weighted_user_attr_item_logits.data.masked_fill_(weighted_attr_item_mask.data, -float('inf'))
        # user_attr_item_output = torch.max(weighted_user_attr_item_logits, 1)[0]

        weighted_attr_item_mask = attr_item_mask
        user_attr_item_output = weighted_user_attr_item_logits.sum(1)/((~weighted_attr_item_mask).sum(1).unsqueeze(-1)
        )
        # """ user """

        attr_user_embed = self.m_attr_embedding(attr_user) 
        attr_user_embed = attr_user_embed.transpose(0, 1)

        attr_user_mask = self.f_generate_mask(attr_lens_user)
    
        attr_attn_user = self.m_attn(attr_user_embed, src_key_padding_mask=attr_user_mask)
        attr_attn_user = attr_attn_user.transpose(0, 1)

        item_embed = self.m_item_embedding(item_ids)

        weighted_item_attr_user_logits = attr_attn_user

        # weighted_attr_user_mask = attr_user_mask.unsqueeze(-1).expand(weighted_item_attr_user_logits.size())
        # weighted_item_attr_user_logits.data.masked_fill_(weighted_attr_user_mask.data, -float('inf'))
        # item_attr_user_output = torch.max(weighted_item_attr_user_logits, 1)[0]

        weighted_attr_user_mask = attr_user_mask
        item_attr_user_output = weighted_item_attr_user_logits.sum(1)/((~weighted_attr_user_mask).sum(1).unsqueeze(-1))

        # user_output = user_embed
        # item_output = item_embed

        user_output = item_attr_user_output
        item_output = user_attr_item_output

        # user_output = (1.0-self.m_gamma)*user_embed+self.m_gamma*item_attr_user_output
        # item_output = (1.0-self.m_gamma)*item_embed+self.m_gamma*user_attr_item_output

        logits_user = torch.matmul(user_output, self.m_output_attr_embedding_user.weight.t())
        logits_item = torch.matmul(item_output, self.m_output_attr_embedding_item.weight.t())

        logits = logits_user+logits_item

        return logits