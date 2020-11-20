"""
data processing for softmax loss
"""

import os
import json
from random import sample
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
from nltk.tokenize import TweetTokenizer
import gensim
import random

class WINE(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_vocab = vocab_obj
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        ###get length
        
        # self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        # self.m_attr_length_user_list = []
        self.m_user_batch_list = []

        self.m_pos_target_list = []
        self.m_pos_len_list = []

        self.m_neg_target_list = []
        self.m_neg_len_list = []
        self.m_negtarget_num_list = []

        self.m_first_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        max_attr_len = args.max_seq_length

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        pos_attr_list = df.pos_attr.tolist()

        pos_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        for (userid_i, itemid_i) in pos_useritem_attrlist_dict:
            pos_attrlist_i = pos_useritem_attrlist_dict[(userid_i, itemid_i)][0]
            pos_attrlist_i = [int(j)+2 for j in pos_attrlist_i]

            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_pos_target_list.append(pos_attrlist_i)
            self.m_pos_len_list.append(len(pos_attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list))
        # exit()

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        item_i = self.m_item_batch_list[i]

        user_i = self.m_user_batch_list[i]

        pos_target_i = self.m_pos_target_list[i]
        pos_len_i = self.m_pos_len_list[i]

        # neg_target_i = self.m_neg_target_list[i]
        # neg_len_i = self.m_neg_len_list[i]
        # negtarget_num_i  = self.m_negtarget_num_list[i]

        # first_i = self.m_first_list[i]

        sample_i = {"user": user_i,  "item": item_i, "pos_target": pos_target_i, "pos_len": pos_len_i, "max_seq_len": self.m_max_seq_len}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)
        item_iter = []
        user_iter = []

        input_target_iter = []
        output_target_iter = []
        output_target_len_iter = []

        pos_len_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

        max_pos_targetlen_iter = max(pos_len_iter)
        # max_neg_targetlen_iter = max(neg_len_iter)
        # max_negtarget_num_iter = max(negtarget_num_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        sos = 1
        eos = 2

        for i in range(batch_size):
            sample_i = batch[i]

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])

            pos_len_i = sample_i["pos_len"]

            input_target_i = [sos]+pos_target_i+[pad_id]*(max_pos_targetlen_iter-pos_len_i)
            input_target_iter.append(input_target_i)

            output_target_i = pos_target_i+[eos]+[pad_id]*(max_pos_targetlen_iter-pos_len_i)
            output_target_iter.append(output_target_i)

            output_target_len_i = pos_len_i+1
            output_target_len_iter.append(output_target_len_i)
            
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        input_target_iter_tensor = torch.from_numpy(np.array(input_target_iter)).long()

        output_target_iter_tensor = torch.from_numpy(np.array(output_target_iter)).long()

        output_target_len_iter_tensor = torch.from_numpy(np.array(output_target_len_iter)).long()

        return input_target_iter_tensor, output_target_iter_tensor, output_target_len_iter_tensor, user_iter_tensor, item_iter_tensor

class WINE_TEST(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_vocab = vocab_obj
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        ###get length
        
        # self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        # self.m_attr_length_user_list = []
        self.m_user_batch_list = []

        self.m_pos_target_list = []
        self.m_pos_len_list = []

        self.m_neg_target_list = []
        self.m_neg_len_list = []
        self.m_negtarget_num_list = []

        self.m_first_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        max_attr_len = args.max_seq_length

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        pos_attr_list = df.pos_attr.tolist()

        pos_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        for (userid_i, itemid_i) in pos_useritem_attrlist_dict:
            pos_attrlist_i = pos_useritem_attrlist_dict[(userid_i, itemid_i)][0]
            pos_attrlist_i = [int(j)+2 for j in pos_attrlist_i]

            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_pos_target_list.append(pos_attrlist_i)
            self.m_pos_len_list.append(len(pos_attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list))
        # exit()

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        item_i = self.m_item_batch_list[i]

        user_i = self.m_user_batch_list[i]

        pos_target_i = self.m_pos_target_list[i]
        pos_len_i = self.m_pos_len_list[i]

        # neg_target_i = self.m_neg_target_list[i]
        # neg_len_i = self.m_neg_len_list[i]
        # negtarget_num_i  = self.m_negtarget_num_list[i]

        # first_i = self.m_first_list[i]

        sample_i = {"user": user_i,  "item": item_i, "pos_target": pos_target_i, "pos_len": pos_len_i, "max_seq_len": self.m_max_seq_len}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)
        item_iter = []
        user_iter = []

        input_target_iter = []
        output_target_iter = []
        output_target_len_iter = []

        pos_len_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

        max_pos_targetlen_iter = max(pos_len_iter)
        # max_neg_targetlen_iter = max(neg_len_iter)
        # max_negtarget_num_iter = max(negtarget_num_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        sos = 1
        eos = 2

        for i in range(batch_size):
            sample_i = batch[i]

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])

            pos_len_i = sample_i["pos_len"]

            input_target_i = [sos]+pos_target_i+[pad_id]*(max_pos_targetlen_iter-pos_len_i)
            input_target_iter.append(input_target_i)

            output_target_i = pos_target_i+[eos]+[pad_id]*(max_pos_targetlen_iter-pos_len_i)
            output_target_iter.append(output_target_i)

            output_target_len_i = pos_len_i+1
            output_target_len_iter.append(output_target_len_i)
            
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        input_target_iter_tensor = torch.from_numpy(np.array(input_target_iter)).long()

        output_target_iter_tensor = torch.from_numpy(np.array(output_target_iter)).long()

        output_target_len_iter_tensor = torch.from_numpy(np.array(output_target_len_iter)).long()

        return input_target_iter_tensor, output_target_iter_tensor, output_target_len_iter_tensor, user_iter_tensor, item_iter_tensor