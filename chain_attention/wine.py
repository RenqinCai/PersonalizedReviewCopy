import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
import random

class WINE(Dataset):
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
        
        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        print("batch num", self.m_batch_num)

        ###get length

        self.m_user_batch_list = []
        self.m_item_batch_list = []

        self.m_input_attr_list = []
        self.m_target_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()

        pos_attr_list = df.pos_attr.tolist()

        # max_attr_len = 20
        max_attr_len = args.max_seq_length

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]

            pos_attr_list_i = pos_attr_list[sample_index]
            pos_attr_list_i = [int(i) for i in pos_attr_list_i]

            random.shuffle(pos_attr_list_i)

            pos_attr_list_len_i = len(pos_attr_list_i)

            if pos_attr_list_len_i == 1:
                continue

            self.m_user_batch_list.append(user_id)
            self.m_item_batch_list.append(item_id)

            self.m_input_attr_list.append([])
            self.m_target_list.append(pos_attr_list_i[0])
               
            for j in range(1, pos_attr_list_len_i):
                input_attr_list_ij = pos_attr_list_i[:j]
                target_ij = pos_attr_list_i[j]
            
                self.m_user_batch_list.append(user_id)
                self.m_item_batch_list.append(item_id)

                self.m_input_attr_list.append(input_attr_list_ij)
                self.m_target_list.append(target_ij)

        print("... load train data ...", len(self.m_item_batch_list), len(self.m_input_attr_list), len(self.m_target_list))

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        user_i = self.m_user_batch_list[i]
       
        item_i = self.m_item_batch_list[i]

        input_attr_list_i = self.m_input_attr_list[i]

        target_i = self.m_target_list[i]

        sample_i = {"user": user_i,  "item": item_i, "input_attr": input_attr_list_i, "target": target_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        user_iter = []
        item_iter = []

        input_attr_iter = []
        input_attr_len_iter = []

        target_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
        
            input_attr_i = sample_i["input_attr"]
            input_attr_len_i = len(input_attr_i)
            input_attr_len_iter.append(input_attr_len_i)

        max_input_attr_len = max(input_attr_len_iter)

        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            user_i = sample_i["user"]
            user_iter.append(user_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)
            
            input_attr_i = copy.deepcopy(sample_i["input_attr"])
            input_attr_len_i = len(input_attr_i)

            input_attr_i.extend([pad_id]*(max_input_attr_len-input_attr_len_i))
            input_attr_iter.append(input_attr_i)

            target_i = sample_i["target"]
            target_iter.append(target_i)

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        input_attr_iter_tensor = torch.from_numpy(np.array(input_attr_iter)).long()

        input_attr_len_iter_tensor = torch.from_numpy(np.array(input_attr_len_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        return  user_iter_tensor, item_iter_tensor, input_attr_iter_tensor, input_attr_len_iter_tensor, target_iter_tensor

class WINE_TEST(Dataset):
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
        
        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        print("batch num", self.m_batch_num)

        ###get length

        self.m_user_batch_list = []
        self.m_item_batch_list = []

        self.m_input_attr_list = []
        self.m_target_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()

        pos_attr_list = df.pos_attr.tolist()

        # max_attr_len = 20
        max_attr_len = args.max_seq_length

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]

            pos_attr_list_i = pos_attr_list[sample_index]
            pos_attr_list_i = [int(i) for i in pos_attr_list_i]

            # random.shuffle(pos_attr_list_i)
            if len(pos_attr_list_i) == 1:
                continue

            self.m_input_attr_list.append([])
            self.m_target_list.append(pos_attr_list_i)

            # pos_attr_list_len_i = len(pos_attr_list_i)
            # for j in range(1, pos_attr_list_len_i):
            #     input_attr_list_ij = pos_attr_list_i[:j]
            #     target_ij = pos_attr_list_i[j]
            
            self.m_user_batch_list.append(user_id)
            self.m_item_batch_list.append(item_id)

            #     self.m_input_attr_list.append(input_attr_list_ij)
            #     self.m_target_list.append(target_ij)

        print("... load train data ...", len(self.m_item_batch_list), len(self.m_input_attr_list), len(self.m_target_list))

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        user_i = self.m_user_batch_list[i]
       
        item_i = self.m_item_batch_list[i]

        input_attr_list_i = self.m_input_attr_list[i]

        target_i = self.m_target_list[i]

        sample_i = {"user": user_i,  "item": item_i, "input_attr": input_attr_list_i, "target": target_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        user_iter = []
        item_iter = []

        # input_attr_iter = []
        # input_attr_len_iter = []

        target_len_iter = []

        target_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
        
            # input_attr_i = sample_i["input_attr"]
            # input_attr_len_i = len(input_attr_i)
            # input_attr_len_iter.append(input_attr_len_i)

            target_i = sample_i["target"]
            target_len_iter.append(len(target_i))

        # max_input_attr_len = max(input_attr_len_iter)
        max_target_attr_len = max(target_len_iter)

        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            user_i = sample_i["user"]
            user_iter.append(user_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)
            
            target_i = copy.deepcopy(sample_i["target"])
            target_len_i = len(target_i)
            target_i.extend([pad_id]*(max_target_attr_len-target_len_i))
            target_iter.append(target_i)

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        target_len_iter_tensor = torch.from_numpy(np.array(target_len_iter)).long()

        # print("target_len_iter_tensor", target_len_iter_tensor)

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        # exit()

        return user_iter_tensor, item_iter_tensor, target_len_iter_tensor, target_iter_tensor

        # return  user_iter_tensor, item_iter_tensor, None, target_iter_tensor