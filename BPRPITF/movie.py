import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
from nltk.tokenize import TweetTokenizer

class _MOVIE(Dataset):
    def __init__(self, args, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        ###get length
        
        self.m_pos_tag_batch_list = []
        self.m_neg_tag_batch_list = []
        self.m_user_batch_list = []
        self.m_item_batch_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        pos_tag_list = df.pos_tagid.tolist()
        neg_tag_list = df.neg_tagid.tolist()

        for sample_index in range(self.m_sample_num):
        # for sample_index in range(1000):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]

            pos_tag_id = pos_tag_list[sample_index]
            neg_tag_id = neg_tag_list[sample_index]
            
            self.m_pos_tag_batch_list.append(pos_tag_id)
            self.m_neg_tag_batch_list.append(neg_tag_id)
            
            # uid = self.m_user2uid[user_id]
            self.m_user_batch_list.append(user_id)

            # iid = self.m_item2iid[item_id]
            self.m_item_batch_list.append(item_id)
        
        print("... load train data ...", len(self.m_pos_tag_batch_list), len(self.m_neg_tag_batch_list), len(self.m_user_batch_list), len(self.m_item_batch_list))
        # exit()

    def __len__(self):
        return len(self.m_user_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx

        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]

        pos_tag_i = self.m_pos_tag_batch_list[i]
        neg_tag_i = self.m_neg_tag_batch_list[i]

        return pos_tag_i, neg_tag_i, user_i, item_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        pos_tag_iter = []
        neg_tag_iter = []
        
        user_iter = []
        item_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            pos_tag_i = copy.deepcopy(sample_i[0])
            pos_tag_iter.append(pos_tag_i)

            neg_tag_i = copy.deepcopy(sample_i[1])
            neg_tag_iter.append(neg_tag_i)

            user_i = sample_i[2]
            user_iter.append(user_i)

            item_i = sample_i[3]
            item_iter.append(item_i)

        pos_tag_iter_tensor = torch.from_numpy(np.array(pos_tag_iter)).long()
        neg_tag_iter_tensor = torch.from_numpy(np.array(neg_tag_iter)).long()
        
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        return pos_tag_iter_tensor, neg_tag_iter_tensor, user_iter_tensor, item_iter_tensor

class _MOVIE_TEST(Dataset):
    def __init__(self, args, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        ###get length
        
        self.m_pos_tag_batch_list = []
        self.m_user_batch_list = []
        self.m_item_batch_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        pos_tag_list = df.tagid.tolist()

        user_item_tag_dict = dict(df.groupby(['userid', 'itemid']).tagid.apply(list))

        for (user_id, item_id) in user_item_tag_dict:
            tag_list = user_item_tag_dict[(user_id, item_id)]            
            self.m_pos_tag_batch_list.append(tag_list)
            
            # uid = self.m_user2uid[user_id]
            self.m_user_batch_list.append(user_id)

            # iid = self.m_item2iid[item_id]
            self.m_item_batch_list.append(item_id)
        
        print("... load train data ...", len(self.m_pos_tag_batch_list), len(self.m_user_batch_list), len(self.m_item_batch_list))
        # exit()

    def __len__(self):
        return len(self.m_user_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx

        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]

        pos_tag_i = self.m_pos_tag_batch_list[i]

        return pos_tag_i, user_i, item_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        pos_tag_iter = []
        mask_iter = []
        len_iter = []
        user_iter = []
        item_iter = []

        max_length = 0
        length_list = []
        for i in range(batch_size):
            sample_i = batch[i]
            pos_tag_i = copy.deepcopy(sample_i[0])
            pos_tag_len_i = len(pos_tag_i)
            length_list.append(pos_tag_len_i)

        max_length = max(length_list)
        # print("max_length", max_length)
        for i in range(batch_size):
            sample_i = batch[i]

            pos_tag_i = copy.deepcopy(sample_i[0])
            len_i = length_list[i]

            # print("len", len_i)
            pos_tag_i.extend([-1]*(max_length-len_i))

            mask_pos_tag_i = pos_tag_i
            # print("mask_pos_tag_i", mask_pos_tag_i)
            pos_tag_iter.append(mask_pos_tag_i)
            mask_iter.append([1]*len_i+[0]*(max_length-len_i))

            user_i = sample_i[1]
            user_iter.append(user_i)

            item_i = sample_i[2]
            item_iter.append(item_i)
        # exit()
        pos_tag_iter_tensor = torch.from_numpy(np.array(pos_tag_iter))
        mask_iter_tensor = torch.from_numpy(np.array(mask_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        return pos_tag_iter_tensor, mask_iter_tensor, user_iter_tensor, item_iter_tensor

