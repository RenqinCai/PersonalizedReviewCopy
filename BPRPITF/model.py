import torch
from torch import log, neg_
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

        self.m_tag_embed_size = args.attr_emb_size
        self.m_user_embed_size = args.user_emb_size
        self.m_item_embed_size = args.item_emb_size

        self.m_tag_user_embedding = nn.Embedding(self.m_vocab_size, self.m_tag_embed_size)
        self.m_tag_item_embedding = nn.Embedding(self.m_vocab_size, self.m_tag_embed_size)
        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        torch.nn.init.normal_(self.m_tag_user_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_tag_item_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_user_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_item_embedding.weight, 0.0, 0.01)


    def forward(self, pos_tag_input, neg_tag_input, user_ids, item_ids):

        ### pos_user_tag_x: batch_size*embedding_size
        pos_user_tag_x = self.m_tag_user_embedding(pos_tag_input)
        
        ### user_x: batch_size*embedding_size
        user_x = self.m_user_embedding(user_ids)

        ### batch_size*1
        # pos_user_tag_score = torch.matmul(pos_user_tag_x, user_x) 
        pos_user_tag_score = pos_user_tag_x*user_x
        pos_user_tag_score = torch.sum(pos_user_tag_score, dim=1)
        
        ### pos_item_tag_x: batch_size*embedding_size
        pos_item_tag_x = self.m_tag_item_embedding(pos_tag_input)

        ### item_x: batch_size*embedding_size
        item_x = self.m_item_embedding(item_ids)

        ### 
        # pos_item_tag_score = torch.matmul(pos_item_tag_x, item_x) 
        pos_item_tag_score = pos_item_tag_x*item_x
        pos_item_tag_score = torch.sum(pos_item_tag_score, dim=1)

        pos_tag_score = pos_user_tag_score + pos_item_tag_score
    
        ### pos_user_tag_x: batch_size*embedding_size
        neg_user_tag_x = self.m_tag_user_embedding(neg_tag_input)
        
        ### batch_size*1
        # neg_user_tag_score = torch.matmul(neg_user_tag_x, user_x) 
        neg_user_tag_score = neg_user_tag_x*user_x
        neg_user_tag_score = torch.sum(neg_user_tag_score, dim=1)
        
        ### pos_item_tag_x: batch_size*embedding_size
        neg_item_tag_x = self.m_tag_item_embedding(neg_tag_input)

        ### 
        # neg_item_tag_score = torch.matmul(neg_item_tag_x, item_x) 
        neg_item_tag_score = neg_item_tag_x*item_x
        neg_item_tag_score = torch.sum(neg_item_tag_score, dim=1)

        neg_tag_score = neg_user_tag_score + neg_item_tag_score

        logits = pos_tag_score-neg_tag_score

        return logits, user_x, item_x, pos_user_tag_x, pos_item_tag_x, neg_user_tag_x, neg_item_tag_x