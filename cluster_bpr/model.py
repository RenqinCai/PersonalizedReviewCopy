"""
softmax as loss
"""

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

        self.m_output_hidden_size = args.output_hidden_size

        self.m_attn_linear_size = args.attn_linear_size

        # self.m_attr_embedding = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        # encoder_layers = TransformerEncoderLayer(self.m_attr_embed_size, self.m_attn_head_num, self.m_attn_linear_size)
        # self.m_attn = TransformerEncoder(encoder_layers, self.m_attn_layer_num)

        self.m_gamma = args.gamma

        # self.m_output_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        # self.m_output_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)

        cluster_num = 256

        self.m_attr_embedding_user = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        self.m_attr_embedding_item = nn.Embedding(self.m_vocab_size, self.m_attr_embed_size)
        
        # print("weight size user", self.m_output_attr_embedding_user.size())
        # self.m_attr_embedding_user.weight = self.m_output_attr_embedding_user.weight
        # self.m_attr_embedding_item.weight = self.m_output_attr_embedding_item.weight

        self.m_dropout = nn.Dropout(p=0.2)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.m_attr_embedding_user.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_attr_embedding_item.weight, -initrange, initrange)

        torch.nn.init.uniform_(self.m_user_embedding.weight, -initrange, initrange)
        torch.nn.init.uniform_(self.m_item_embedding.weight, -initrange, initrange)

    def f_generate_mask(self, length):
        max_len = length.max().item()
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        mask = ~mask

        return mask

    def f_get_logits(self, embed, attr):
        logits = torch.matmul(embed, attr.unsqueeze(-1))
        logits = logits.squeeze(-1)

        return logits

    def forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids):

        ### attr_item_mask: batch_size*max_len
        attr_item_mask = self.f_generate_mask(attr_lens_item)
        attr_item_mask = ~attr_item_mask

        ### attr_embed_item: batch_size*max_len*embed_size
        attr_embed_item = self.m_attr_embedding_item(attr_item)

        # ### cluster_weight_item: batch_size*max_len*cluster_num
        cluster_weight_item = torch.matmul(attr_embed_item, self.m_attr_embedding_item.weight.t())

        # ### cluster_weight_item: batch_size*max_len*cluster_num
        cluster_weight_item = torch.softmax(cluster_weight_item, dim=1)

        # ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_item_cluster = (attr_embed_item.unsqueeze(2) * cluster_weight_item.unsqueeze(-1))

        # ### attr_tf_item: batch_size*max_len
        ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_item_cluster = attr_embed_item_cluster*((attr_tf_item*attr_item_mask).unsqueeze(-1).unsqueeze(-1))

        # # ### attr_embed_item_cluster: batch_size*cluster_num*embed_size
        attr_embed_item_cluster = torch.sum(attr_embed_item_cluster, dim=1)
        
        ### logits_item_cluster: batch_size*voc_size*embed_size
        logits_item_cluster = attr_embed_item_cluster*(self.m_attr_embedding_item.weight.unsqueeze(0))

        ### logits_item_cluster: batch_size*voc_size
        logits_item_cluster = torch.sum(logits_item_cluster, dim=-1)

        # ### logits_item_cluster: batch_size*cluster_num*voc_size
        # logits_item_cluster = torch.matmul(attr_embed_item_cluster, self.m_output_attr_embedding_item.t())

        # ### logits_item_cluster: batch_size*voc_size
        # logits_item_cluster, _ = torch.max(logits_item_cluster, dim=1)

        """"""

        ### attr_user_mask: batch_size*max_len
        attr_user_mask = self.f_generate_mask(attr_lens_user)
        attr_user_mask = ~attr_user_mask

        ### attr_embed_user: batch_size*max_len*embed_size
        attr_embed_user = self.m_attr_embedding_user(attr_user)

        ### cluster_weight_item: batch_size*max_len*cluster_num
        cluster_weight_user = torch.matmul(attr_embed_user, self.m_attr_embedding_user.weight.t())

        cluster_weight_user = torch.softmax(cluster_weight_user, dim=1)

        ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_user_cluster = (attr_embed_user.unsqueeze(2) * cluster_weight_user.unsqueeze(-1))

        ### attr_tf_user: batch_size*max_len
        attr_embed_user_cluster = attr_embed_user_cluster*((attr_tf_user*attr_user_mask).unsqueeze(-1).unsqueeze(-1))

        ### attr_embed_item_cluster: batch_size*cluster_num*embed_size
        attr_embed_user_cluster = torch.sum(attr_embed_user_cluster, dim=1)

        ### logits_item_cluster: batch_size*voc_size*embed_size
        logits_user_cluster = attr_embed_user_cluster*(self.m_attr_embedding_user.weight.unsqueeze(0))

        ### logits_item_cluster: batch_size*voc_size
        logits_user_cluster = torch.sum(logits_user_cluster, dim=-1)

        # ### logits_item_cluster: batch_size*cluster_num*voc_size
        # logits_user_cluster = torch.matmul(attr_embed_user_cluster, self.m_output_attr_embedding_user.t())

        # ### logits_user_cluster: batch_size*voc_size
        # logits_user_cluster, _ = torch.max(logits_user_cluster, dim=1)

        logits = logits_item_cluster+logits_user_cluster

        user_embed = self.m_user_embedding(user_ids)

        item_embed = self.m_item_embedding(item_ids)

        attr_embed_user = self.m_attr_embedding_user.weight

        attr_embed_item = self.m_attr_embedding_item.weight

        logits_user = torch.matmul(user_embed, attr_embed_user.t())
        logits_item = torch.matmul(item_embed, attr_embed_item.t())

        logits = logits+logits_user+logits_item

        return logits, None, None

    def f_eval_forward(self, attr_item, attr_tf_item, attr_lens_item, item_ids, attr_user, attr_tf_user, attr_lens_user, user_ids):

         ### attr_item_mask: batch_size*max_len
        attr_item_mask = self.f_generate_mask(attr_lens_item)
        attr_item_mask = ~attr_item_mask

        ### attr_embed_item: batch_size*max_len*embed_size
        attr_embed_item = self.m_attr_embedding_item(attr_item)

        # ### cluster_weight_item: batch_size*max_len*cluster_num
        cluster_weight_item = torch.matmul(attr_embed_item, self.m_attr_embedding_item.weight.t())

        cluster_weight_item = torch.softmax(cluster_weight_item, dim=1)

        # ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_item_cluster = (attr_embed_item.unsqueeze(2) * cluster_weight_item.unsqueeze(-1))

        # ### attr_tf_item: batch_size*max_len
        ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_item_cluster = attr_embed_item_cluster*((attr_tf_item*attr_item_mask).unsqueeze(-1).unsqueeze(-1))

        # # ### attr_embed_item_cluster: batch_size*cluster_num*embed_size
        attr_embed_item_cluster = torch.sum(attr_embed_item_cluster, dim=1)
        
        ### logits_item_cluster: batch_size*voc_size*embed_size
        logits_item_cluster = attr_embed_item_cluster*(self.m_attr_embedding_item.weight.unsqueeze(0))

        ### logits_item_cluster: batch_size*voc_size
        logits_item_cluster = torch.sum(logits_item_cluster, dim=-1)

        # ### logits_item_cluster: batch_size*cluster_num*voc_size
        # logits_item_cluster = torch.matmul(attr_embed_item_cluster, self.m_output_attr_embedding_item.t())

        # ### logits_item_cluster: batch_size*voc_size
        # logits_item_cluster, _ = torch.max(logits_item_cluster, dim=1)

        """"""

        ### attr_user_mask: batch_size*max_len
        attr_user_mask = self.f_generate_mask(attr_lens_user)
        attr_user_mask = ~attr_user_mask

        ### attr_embed_user: batch_size*max_len*embed_size
        attr_embed_user = self.m_attr_embedding_user(attr_user)

        ### cluster_weight_item: batch_size*max_len*cluster_num
        cluster_weight_user = torch.matmul(attr_embed_user, self.m_attr_embedding_user.weight.t())

        cluster_weight_user = torch.softmax(cluster_weight_user, dim=1)

        ### attr_embed_item_cluster: batch_size*max_len*cluster_num*embed_size
        attr_embed_user_cluster = (attr_embed_user.unsqueeze(2) * cluster_weight_user.unsqueeze(-1))

        ### attr_tf_user: batch_size*max_len
        attr_embed_user_cluster = attr_embed_user_cluster*((attr_tf_user*attr_user_mask).unsqueeze(-1).unsqueeze(-1))

        ### attr_embed_item_cluster: batch_size*cluster_num*embed_size
        attr_embed_user_cluster = torch.sum(attr_embed_user_cluster, dim=1)

        ### logits_item_cluster: batch_size*voc_size*embed_size
        logits_user_cluster = attr_embed_user_cluster*(self.m_attr_embedding_user.weight.unsqueeze(0))

        ### logits_item_cluster: batch_size*voc_size
        logits_user_cluster = torch.sum(logits_user_cluster, dim=-1)

        # ### logits_item_cluster: batch_size*cluster_num*voc_size
        # logits_user_cluster = torch.matmul(attr_embed_user_cluster, self.m_output_attr_embedding_user.t())

        # ### logits_user_cluster: batch_size*voc_size
        # logits_user_cluster, _ = torch.max(logits_user_cluster, dim=1)

        logits = logits_item_cluster+logits_user_cluster

        user_embed = self.m_user_embedding(user_ids)

        item_embed = self.m_item_embedding(item_ids)

        attr_embed_user = self.m_attr_embedding_user.weight

        attr_embed_item = self.m_attr_embedding_item.weight

        logits_user = torch.matmul(user_embed, attr_embed_user.t())
        logits_item = torch.matmul(item_embed, attr_embed_item.t())

        logits = logits+logits_user+logits_item

        # print("lgoits", logits.size())

        return logits

