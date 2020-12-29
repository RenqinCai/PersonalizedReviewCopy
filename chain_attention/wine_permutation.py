"""
permutate attributes
"""

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
    
        ###get length

        self.m_user_batch_list = []
        self.m_item_batch_list = []

        self.m_input_attr_list = []
        self.m_target_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()

        # print("user_num", df.userid.nunique())
        # print("item_num", df.itemid.nunique())

        pos_attr_list = df.pos_attr.tolist()
        max_attr_len = args.max_seq_length

        df_size = len(df)

        for sample_index in range(df_size):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]

            pos_attr_list_i = pos_attr_list[sample_index]
            pos_attr_list_i = [int(i) for i in pos_attr_list_i]

            # pos_attr_list_len_i = len(pos_attr_list_i)

            # if pos_attr_list_len_i == 1:
            #     continue

            self.generate_sample_ind(user_id, item_id, pos_attr_list_i)
            # # self.m_input_attr_list.append([])
            # self.m_target_list.append(pos_attr_list_i[0])
               
            # for j in range(1, pos_attr_list_len_i):
            #     input_attr_list_ij = pos_attr_list_i[:j]
            #     target_ij = pos_attr_list_i[j]
            
            #     self.m_user_batch_list.append(user_id)
            #     self.m_item_batch_list.append(item_id)

            #     self.m_input_attr_list.append(input_attr_list_ij)
            #     self.m_target_list.append(target_ij)

        print("... load train data ...", len(self.m_item_batch_list), len(self.m_input_attr_list), len(self.m_target_list))

        self.m_sample_num = len(self.m_item_batch_list)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        print("batch num", self.m_batch_num)

    def generate_sample_ind(self, user_id, item_id, pos_attr_list_i):
        pos_attr_list_len_i = len(pos_attr_list_i)
        # if pos_attr_list_len_i == 1:
        #     return
        
        shuffle_freq = 1
        shuffle_freq = 2
        shuffle_freq = 3
        shuffle_freq = 4

        for shuffle_i in range(shuffle_freq):
            random.shuffle(pos_attr_list_i)

        for j in range(pos_attr_list_len_i):
            input_attr_list_ij = list(pos_attr_list_i[:j])
            target_ij = pos_attr_list_i[j]

            self.m_user_batch_list.append(user_id)
            self.m_item_batch_list.append(item_id)

            self.m_input_attr_list.append(input_attr_list_ij)
            self.m_target_list.append(target_ij)

        # print(pos_attr_list_i)
        # pos_attr_list_i = np.array(pos_attr_list_i)
        # permutation_num = 100
        # len_threshold = 9
        # if pos_attr_list_len_i < len_threshold:
        #     # print("len_threshold", len_threshold)
        #     mask_list = [[0], [1]]
        #     for i in range(1, pos_attr_list_len_i):
        #         next_mask_list = []
        #         for tmp in mask_list:
        #             tmp_0 = copy.deepcopy(tmp)
        #             tmp_0.append(0)
        #             next_mask_list.append(tmp_0)

        #             tmp_1 = copy.deepcopy(tmp)
        #             tmp_1.append(1)
        #             next_mask_list.append(tmp_1)

        #         mask_list = next_mask_list

        #     mask_list = mask_list[:-1]
        #     # print("mask_list", mask_list)
        #     random.shuffle(mask_list)
        #     # print("shuffle mask_list", mask_list)
        #     input_mask_list = mask_list[:permutation_num]
        #     for mask_index in input_mask_list:
        #         mask_index = np.array(mask_index)==1
        #         # print("mask_index", mask_index)
        #         # print("~mask_index", ~mask_index)
        #         input_attr_list_ij = list(pos_attr_list_i[mask_index])
        #         target_ij = list(pos_attr_list_i[~mask_index])
        #         if len(target_ij) == 0:
        #             continue
        #         target_ij = random.sample(target_ij, 1)[0]
        #         # input_attr_list_ij = mask_attr_list_ij[:-1]
        #         # target_ij = mask_attr_list_ij[-1]

        #         self.m_user_batch_list.append(user_id)
        #         self.m_item_batch_list.append(item_id)

        #         self.m_input_attr_list.append(input_attr_list_ij)
        #         self.m_target_list.append(target_ij)

        #         # print("input_attr_list_ij", input_attr_list_ij)
        #         # print("target_ij", target_ij)
        # else:
        #     for j in range(permutation_num):
        #         mask_index = np.array([random.randint(0, 1)==1 for k in range(pos_attr_list_len_i)])
                
        #         # mask_attr_list_ij = list(pos_attr_list_i[mask_index])
        #         # if len(mask_attr_list_ij) == 0:
        #         #     continue

        #         input_attr_list_ij = list(pos_attr_list_i[mask_index])
        #         target_ij = list(pos_attr_list_i[~mask_index])
        #         if len(target_ij) == 0:
        #             continue
        #         target_ij = random.sample(target_ij, 1)[0]

        #         # input_attr_list_ij = mask_attr_list_ij[:-1]
        #         # target_ij = mask_attr_list_ij[-1]
        #         # target_ij = list(pos_attr_list_i[~mask_index])

        #         self.m_user_batch_list.append(user_id)
        #         self.m_item_batch_list.append(item_id)

        #         self.m_input_attr_list.append(input_attr_list_ij)
        #         self.m_target_list.append(target_ij)

    def generate_sample_group(self, user_id, item_id, pos_attr_list_i):
        pos_attr_list_len_i = len(pos_attr_list_i)
        if pos_attr_list_len_i == 1:
            return
        
        # print(pos_attr_list_i)
        pos_attr_list_i = np.array(pos_attr_list_i)
        permutation_num = 100
        len_threshold = 9
        if pos_attr_list_len_i < len_threshold:
            # print("len_threshold", len_threshold)
            mask_list = [[0], [1]]
            for i in range(1, pos_attr_list_len_i):
                next_mask_list = []
                for tmp in mask_list:
                    tmp_0 = copy.deepcopy(tmp)
                    tmp_0.append(0)
                    next_mask_list.append(tmp_0)

                    tmp_1 = copy.deepcopy(tmp)
                    tmp_1.append(1)
                    next_mask_list.append(tmp_1)

                mask_list = next_mask_list

            mask_list = mask_list[:-1]
            # print("mask_list", mask_list)
            random.shuffle(mask_list)
            # print("shuffle mask_list", mask_list)
            input_mask_list = mask_list[:permutation_num]
            for mask_index in input_mask_list:
                mask_index = np.array(mask_index)==1
                # print("mask_index", mask_index)
                # print("~mask_index", ~mask_index)
                input_attr_list_ij = list(pos_attr_list_i[mask_index])
                target_ij = list(pos_attr_list_i[~mask_index])

                self.m_user_batch_list.append(user_id)
                self.m_item_batch_list.append(item_id)

                self.m_input_attr_list.append(input_attr_list_ij)
                self.m_target_list.append(target_ij)

                # print("input_attr_list_ij", input_attr_list_ij)
                # print("target_ij", target_ij)
        else:
            for j in range(permutation_num):
                mask_index = np.array([random.randint(0, 1)==1 for k in range(pos_attr_list_len_i)])
                
                input_attr_list_ij = list(pos_attr_list_i[mask_index])
                target_ij = list(pos_attr_list_i[~mask_index])

                if len(target_ij) == 0:
                    continue

                self.m_user_batch_list.append(user_id)
                self.m_item_batch_list.append(item_id)

                self.m_input_attr_list.append(input_attr_list_ij)
                self.m_target_list.append(target_ij)

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
        target_len_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
        
            input_attr_i = sample_i["input_attr"]
            input_attr_len_i = len(input_attr_i)
            input_attr_len_iter.append(input_attr_len_i)

            # target_len_i = len(sample_i["target"])
            # target_len_iter.append(target_len_i)

        max_input_attr_len = max(input_attr_len_iter)
        # max_target_len = max(target_len_iter)

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

            target_i = copy.deepcopy(sample_i["target"])
            # print("target_i", target_i)
            # target_len_i = len(target_i)
            # target_i.extend([pad_id]*(max_target_len-target_len_i))
            target_iter.append(target_i)

        # exit()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        input_attr_iter_tensor = torch.from_numpy(np.array(input_attr_iter)).long()

        input_attr_len_iter_tensor = torch.from_numpy(np.array(input_attr_len_iter)).long()

        target_len_iter_tensor = torch.from_numpy(np.array(target_len_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        return  user_iter_tensor, item_iter_tensor, input_attr_iter_tensor, input_attr_len_iter_tensor, target_iter_tensor, target_len_iter_tensor

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
            # if len(pos_attr_list_i) == 1:
            #     continue

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