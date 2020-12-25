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
import gensim
import random

class MOVIE(Dataset):
    def __init__(self, args, vocab_obj, df, boa_item_dict, boa_user_dict):
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
        
        self.m_attr_item_list = []
        self.m_attr_tf_item_list = []
        self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        self.m_attr_user_list = []
        self.m_attr_tf_user_list = []
        self.m_attr_length_user_list = []
        self.m_user_batch_list = []
        
        self.m_target_list = []
        self.m_target_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        
        attr_list = df.pos_attr.tolist()

        # max_attr_len = 20
        # max_attr_len = 100
        max_attr_len = args.max_seq_length
        
        # print(list(boa_item_dict.keys()))

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])
            attrlist_i = [int(j) for j in attrlist_i]
            # print("itemid", item_id)
                   
            self.m_item_batch_list.append(item_id)
            
          
            self.m_user_batch_list.append(user_id)

            self.m_target_list.append(attrlist_i)
            self.m_target_len_list.append(len(attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list), len(self.m_attr_tf_item_list), len(self.m_attr_user_list), len(self.m_attr_tf_user_list))
        # exit()

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        
        item_i = self.m_item_batch_list[i]

       
        user_i = self.m_user_batch_list[i]

        target_i = self.m_target_list[i]
        target_len_i = self.m_target_len_list[i]

        sample_i = {"item": item_i,  "user": user_i,  "target": target_i, "target_len": target_len_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        attr_item_iter = []
        attr_tf_item_iter = []
        attr_length_item_iter = []
        item_iter = []

        attr_user_iter = []
        attr_tf_user_iter = []
        attr_length_user_iter = []
        user_iter = []

        target_iter = []
        target_len_iter = []
        target_mask_iter = []

        # attr_item_user_iter = []
        # attr_length_item_user_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
          
            target_len_i = sample_i["target_len"]
            target_len_iter.append(target_len_i)

        
            # attr_item_user_i = set(attr_item_i).union(set(attr_user_i))
            # attr_item_user_i = list(attr_item_user_i)

            # attr_item_user_iter.append(attr_item_user_i)
            # attr_length_item_user_iter.append(len(attr_item_user_i))

        # max_attr_length_item_iter = max(attr_length_item_iter)
        # max_attr_length_user_iter = max(attr_length_user_iter)
        # max_attr_length_item_user_iter = max(attr_length_item_user_iter)

        max_targetlen_iter = max(target_len_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            # attr_item_i = copy.deepcopy(sample_i["attr_item"])
            # attr_user_i = copy.deepcopy(sample_i["attr_user"])

            # attr_item_i = [int(i) for i in attr_item_i]
            # attr_user_i = [int(i) for i in attr_user_i]

            # attr_item_user = set(attr_item_i).union(set(attr_user_i))
            # attr_item_user = list(attr_item_user)

            # attr_item_i = attr_item_iter[i]
            # attr_user_i = attr_user_iter[i]
            # # attr_item_user_i = attr_item_user_iter[i]
            # attr_length_item_user_i = attr_length_item_user_iter[i]

            # attr_item_user_i.extend([pad_id]*(max_attr_length_item_user_iter-attr_length_item_user_i))
            
            # attr_tf_item_i = copy.deepcopy(sample_i['attr_tf_item'])
            # attr_length_item_i = sample_i["attr_length_item"]
            
            # attr_item_i.extend([pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            # # attr_item_iter.append(attr_item_i)

            # attr_tf_item_i.extend([freq_pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            # attr_tf_item_iter.append(attr_tf_item_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)

            # attr_tf_user_i = copy.deepcopy(sample_i['attr_tf_user'])
            # attr_length_user_i = sample_i["attr_length_user"]

            # attr_user_i.extend([pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            # # attr_user_iter.append(attr_user_i)

            # attr_tf_user_i.extend([freq_pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            # attr_tf_user_iter.append(attr_tf_user_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            target_i = copy.deepcopy(sample_i["target"])

            target_len_i = sample_i["target_len"]
            target_i.extend([pad_id]*(max_targetlen_iter-target_len_i))
            target_iter.append(target_i)

            target_mask_iter.append([1]*target_len_i+[0]*(max_targetlen_iter-target_len_i))
            
        attr_item_iter_tensor = torch.from_numpy(np.array(attr_item_iter)).long()
        attr_tf_item_iter_tensor = torch.from_numpy(np.array(attr_tf_item_iter)).float()
        attr_length_item_iter_tensor = torch.from_numpy(np.array(attr_length_item_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        attr_user_iter_tensor = torch.from_numpy(np.array(attr_user_iter)).long()
        attr_tf_user_iter_tensor = torch.from_numpy(np.array(attr_tf_user_iter)).float()
        attr_length_user_iter_tensor = torch.from_numpy(np.array(attr_length_user_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
        target_mask_iter_tensor = torch.from_numpy(np.array(target_mask_iter)).long()

        # samples_tensor = torch.from_numpy(np.array(attr_item_user_iter)).long()

        return attr_item_iter_tensor, attr_tf_item_iter_tensor, attr_length_item_iter_tensor, item_iter_tensor, attr_user_iter_tensor, attr_tf_user_iter_tensor, attr_length_user_iter_tensor, user_iter_tensor, target_iter_tensor, target_mask_iter_tensor

class MOVIE_TEST(Dataset):
    def __init__(self, args, vocab_obj, df, boa_item_dict, boa_user_dict):
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
        
        self.m_attr_item_list = []
        self.m_attr_tf_item_list = []
        self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        self.m_attr_user_list = []
        self.m_attr_tf_user_list = []
        self.m_attr_length_user_list = []
        self.m_user_batch_list = []
        
        self.m_target_list = []
        self.m_target_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        
        attr_list = df.pos_attr.tolist()

        # max_attr_len = 20
        # max_attr_len = 100
        max_attr_len = args.max_seq_length
        
        # print(list(boa_item_dict.keys()))

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])
            attrlist_i = [int(j) for j in attrlist_i]
            # print("itemid", item_id)
                   
            self.m_item_batch_list.append(item_id)
            
          
            self.m_user_batch_list.append(user_id)

            self.m_target_list.append(attrlist_i)
            self.m_target_len_list.append(len(attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list), len(self.m_attr_tf_item_list), len(self.m_attr_user_list), len(self.m_attr_tf_user_list))
        # exit()

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        
        item_i = self.m_item_batch_list[i]

       
        user_i = self.m_user_batch_list[i]

        target_i = self.m_target_list[i]
        target_len_i = self.m_target_len_list[i]

        sample_i = {"item": item_i,  "user": user_i,  "target": target_i, "target_len": target_len_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        attr_item_iter = []
        attr_tf_item_iter = []
        attr_length_item_iter = []
        item_iter = []

        attr_user_iter = []
        attr_tf_user_iter = []
        attr_length_user_iter = []
        user_iter = []

        target_iter = []
        target_len_iter = []
        target_mask_iter = []

        # attr_item_user_iter = []
        # attr_length_item_user_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
          
            target_len_i = sample_i["target_len"]
            target_len_iter.append(target_len_i)

        
            # attr_item_user_i = set(attr_item_i).union(set(attr_user_i))
            # attr_item_user_i = list(attr_item_user_i)

            # attr_item_user_iter.append(attr_item_user_i)
            # attr_length_item_user_iter.append(len(attr_item_user_i))

        # max_attr_length_item_iter = max(attr_length_item_iter)
        # max_attr_length_user_iter = max(attr_length_user_iter)
        # max_attr_length_item_user_iter = max(attr_length_item_user_iter)

        max_targetlen_iter = max(target_len_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            # attr_item_i = copy.deepcopy(sample_i["attr_item"])
            # attr_user_i = copy.deepcopy(sample_i["attr_user"])

            # attr_item_i = [int(i) for i in attr_item_i]
            # attr_user_i = [int(i) for i in attr_user_i]

            # attr_item_user = set(attr_item_i).union(set(attr_user_i))
            # attr_item_user = list(attr_item_user)

            # attr_item_i = attr_item_iter[i]
            # attr_user_i = attr_user_iter[i]
            # # attr_item_user_i = attr_item_user_iter[i]
            # attr_length_item_user_i = attr_length_item_user_iter[i]

            # attr_item_user_i.extend([pad_id]*(max_attr_length_item_user_iter-attr_length_item_user_i))
            
            # attr_tf_item_i = copy.deepcopy(sample_i['attr_tf_item'])
            # attr_length_item_i = sample_i["attr_length_item"]
            
            # attr_item_i.extend([pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            # # attr_item_iter.append(attr_item_i)

            # attr_tf_item_i.extend([freq_pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            # attr_tf_item_iter.append(attr_tf_item_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)

            # attr_tf_user_i = copy.deepcopy(sample_i['attr_tf_user'])
            # attr_length_user_i = sample_i["attr_length_user"]

            # attr_user_i.extend([pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            # # attr_user_iter.append(attr_user_i)

            # attr_tf_user_i.extend([freq_pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            # attr_tf_user_iter.append(attr_tf_user_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            target_i = copy.deepcopy(sample_i["target"])

            target_len_i = sample_i["target_len"]
            target_i.extend([pad_id]*(max_targetlen_iter-target_len_i))
            target_iter.append(target_i)

            target_mask_iter.append([1]*target_len_i+[0]*(max_targetlen_iter-target_len_i))
            
        attr_item_iter_tensor = torch.from_numpy(np.array(attr_item_iter)).long()
        attr_tf_item_iter_tensor = torch.from_numpy(np.array(attr_tf_item_iter)).float()
        attr_length_item_iter_tensor = torch.from_numpy(np.array(attr_length_item_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        attr_user_iter_tensor = torch.from_numpy(np.array(attr_user_iter)).long()
        attr_tf_user_iter_tensor = torch.from_numpy(np.array(attr_tf_user_iter)).float()
        attr_length_user_iter_tensor = torch.from_numpy(np.array(attr_length_user_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
        target_mask_iter_tensor = torch.from_numpy(np.array(target_mask_iter)).long()

        # samples_tensor = torch.from_numpy(np.array(attr_item_user_iter)).long()

        return attr_item_iter_tensor, attr_tf_item_iter_tensor, attr_length_item_iter_tensor, item_iter_tensor, attr_user_iter_tensor, attr_tf_user_iter_tensor, attr_length_user_iter_tensor, user_iter_tensor, target_iter_tensor, target_mask_iter_tensor
