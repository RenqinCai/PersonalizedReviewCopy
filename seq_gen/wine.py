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

class _WINE(Dataset):
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
        neg_attr_list = df.neg_attr.tolist()
        first_list = df.first_flag.tolist()

        # neg_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).neg_attr.apply(list))
        # pos_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        # pair_num = len(neg_useritem_attrlist_dict)

        # print("neg pair", len(neg_useritem_attrlist_dict))

        for sample_index in range(self.m_sample_num):
            userid_i = userid_list[sample_index]
            itemid_i = itemid_list[sample_index]

            pos_attrlist_i = list(pos_attr_list[sample_index])
            pos_attrlist_i = [int(j) for j in pos_attrlist_i]

        # for (userid_i, itemid_i) in pos_useritem_attrlist_dict:
            # pos_attrlist_i = pos_useritem_attrlist_dict[(userid_i, itemid_i)][0]
            # pos_attrlist_i = [int(j) for j in pos_attrlist_i]

            # neg_attrlist_i = neg_useritem_attrlist_dict[(userid_i, itemid_i)]

            neg_attrlist_list_i = list(neg_attr_list[sample_index])
            # neg_attrlist_i = [int(j) for j in neg_attrlist_i]

            neg_target_list_i = []
            neg_len_list_i = []

            first_i = first_list[sample_index]

            # neg_target_list_i = [ [k for k in j] for j in neg_attrlist_i]
            # neg_len_list_i = [len(j) for j in neg_attrlist_i]

            for j in neg_attrlist_list_i:
                j = [int(k) for k in j]
                neg_target_list_i.append(j)
                neg_len_list_i.append(len(j))

            self.m_neg_target_list.append(neg_target_list_i)
            self.m_neg_len_list.append(neg_len_list_i)

            self.m_negtarget_num_list.append(len(neg_target_list_i))
        
            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_pos_target_list.append(pos_attrlist_i)
            self.m_pos_len_list.append(len(pos_attrlist_i))

            self.m_first_list.append(first_i)

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

        neg_target_i = self.m_neg_target_list[i]
        neg_len_i = self.m_neg_len_list[i]
        negtarget_num_i  = self.m_negtarget_num_list[i]

        first_i = self.m_first_list[i]

        sample_i = {"user": user_i,  "item": item_i, "pos_target": pos_target_i, "pos_len": pos_len_i, "neg_target": neg_target_i, "neg_len": neg_len_i, "negtarget_num": negtarget_num_i, "max_seq_len": self.m_max_seq_len, "first": first_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)
        item_iter = []
        user_iter = []

        pos_target_iter = []
        pos_len_iter = []

        neg_target_iter = []
        neg_len_iter = []
        negtarget_num_iter = []

        first_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

            neg_len_i = sample_i["neg_len"]
            for j in neg_len_i:
                neg_len_iter.append(j)

            negtarget_num_i = sample_i["negtarget_num"]
            negtarget_num_iter.append(negtarget_num_i)

        max_pos_targetlen_iter = max(pos_len_iter)
        max_neg_targetlen_iter = max(neg_len_iter)
        max_negtarget_num_iter = max(negtarget_num_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])
            pos_len_i = sample_i["pos_len"]
            pos_target_i.extend([pad_id]*(max_pos_targetlen_iter-pos_len_i))
            pos_target_iter.append(pos_target_i)
            
            neg_target_i = copy.deepcopy(sample_i["neg_target"])
            neg_len_i = sample_i["neg_len"]

            negtarget_num_i = sample_i["negtarget_num"]
            for j in range(negtarget_num_i):
                neg_target_ij = neg_target_i[j]
                neg_len_ij = neg_len_i[j]
                neg_target_ij.extend([pad_id]*(max_neg_targetlen_iter-neg_len_ij))
                neg_target_iter.append(neg_target_ij)
            # neg_target_i.extend([pad_id]*(max_neg_targetlen_iter-neg_len_i)

            first_i = sample_i["first"]
            first_iter.append(first_i)

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        pos_target_iter_tensor = torch.from_numpy(np.array(pos_target_iter)).long()
        pos_len_iter_tensor = torch.from_numpy(np.array(pos_len_iter)).long()

        neg_target_iter_tensor = torch.from_numpy(np.array(neg_target_iter)).long()
        neg_len_iter_tensor = torch.from_numpy(np.array(neg_len_iter)).long()
        negtarget_num_iter_tensor = torch.from_numpy(np.array(negtarget_num_iter)).long()

        first_iter_tensor = torch.from_numpy(np.array(first_iter)).bool()

        return pos_target_iter_tensor, pos_len_iter_tensor, neg_target_iter_tensor, neg_len_iter_tensor, negtarget_num_iter_tensor, user_iter_tensor, item_iter_tensor, first_iter_tensor

class _WINE_TEST(Dataset):
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
        neg_attr_list = df.neg_attr.tolist()
        first_list = df.first_flag.tolist()
        # neg_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).neg_attr.apply(list))
        # pos_useritem_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        # pair_num = len(neg_useritem_attrlist_dict)

        # print("neg pair", len(neg_useritem_attrlist_dict))
        # print("pos pair", len(pos_useritem_attrlist_dict))

        for sample_index in range(self.m_sample_num):
        
            userid_i = userid_list[sample_index]
            itemid_i = itemid_list[sample_index]

            pos_attrlist_i = list(pos_attr_list[sample_index])
            pos_attrlist_i = [int(j) for j in pos_attrlist_i]

            first_i = first_list[sample_index]
        # for (userid_i, itemid_i) in pos_useritem_attrlist_dict:
        #     pos_attrlist_i = pos_useritem_attrlist_dict[(userid_i, itemid_i)][0]
        #     pos_attrlist_i = [int(j) for j in pos_attrlist_i]

        #     neg_attrlist_i = neg_useritem_attrlist_dict[(userid_i, itemid_i)]
    
            neg_attrlist_i = list(neg_attr_list[sample_index])
            # neg_attrlist_i = [int(j) for j in neg_attrlist_i]

            neg_target_list_i = []
            neg_len_list_i = []

            for j in neg_attrlist_i:
                j = [int(k) for k in j]
                neg_target_list_i.append(j)
                neg_len_list_i.append(len(j))

            self.m_neg_target_list.append(neg_target_list_i)
            self.m_neg_len_list.append(neg_len_list_i)

            self.m_negtarget_num_list.append(len(neg_attrlist_i))
        
            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_pos_target_list.append(pos_attrlist_i)
            self.m_pos_len_list.append(len(pos_attrlist_i))

            self.m_first_list.append(first_i)

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

        neg_target_i = self.m_neg_target_list[i]
        neg_len_i = self.m_neg_len_list[i]
        negtarget_num_i  = self.m_negtarget_num_list[i]

        first_i = self.m_first_list[i]

        sample_i = {"user": user_i,  "item": item_i, "pos_target": pos_target_i, "pos_len": pos_len_i, "neg_target": neg_target_i, "neg_len": neg_len_i, "negtarget_num": negtarget_num_i, "max_seq_len": self.m_max_seq_len, "first": first_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)
        item_iter = []
        user_iter = []

        pos_target_iter = []
        pos_len_iter = []

        neg_target_iter = []
        neg_len_iter = []
        negtarget_num_iter = []

        first_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

            neg_len_i = sample_i["neg_len"]
            for j in neg_len_i:
                neg_len_iter.append(j)

            negtarget_num_i = sample_i["negtarget_num"]
            negtarget_num_iter.append(negtarget_num_i)

        max_pos_targetlen_iter = max(pos_len_iter)
        max_neg_targetlen_iter = max(neg_len_iter)
        max_negtarget_num_iter = max(negtarget_num_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])
            pos_len_i = sample_i["pos_len"]
            pos_target_i.extend([pad_id]*(max_pos_targetlen_iter-pos_len_i))
            pos_target_iter.append(pos_target_i)
            
            neg_target_i = copy.deepcopy(sample_i["neg_target"])
            neg_len_i = sample_i["neg_len"]

            negtarget_num_i = sample_i["negtarget_num"]
            for j in range(negtarget_num_i):
                neg_target_ij = neg_target_i[j]
                neg_len_ij = neg_len_i[j]
                neg_target_ij.extend([pad_id]*(max_neg_targetlen_iter-neg_len_ij))
                neg_target_iter.append(neg_target_ij)

            first_i = sample_i["first"]
            first_iter.append(first_i)

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        pos_target_iter_tensor = torch.from_numpy(np.array(pos_target_iter)).long()
        pos_len_iter_tensor = torch.from_numpy(np.array(pos_len_iter)).long()

        neg_target_iter_tensor = torch.from_numpy(np.array(neg_target_iter)).long()
        neg_len_iter_tensor = torch.from_numpy(np.array(neg_len_iter)).long()
        negtarget_num_iter_tensor = torch.from_numpy(np.array(negtarget_num_iter)).long()

        first_iter_tensor = torch.from_numpy(np.array(first_iter)).bool()

        return pos_target_iter_tensor, pos_len_iter_tensor, neg_target_iter_tensor, neg_len_iter_tensor, negtarget_num_iter_tensor, user_iter_tensor, item_iter_tensor, first_iter_tensor