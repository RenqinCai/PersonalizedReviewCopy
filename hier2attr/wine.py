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

class _WINE(Dataset):
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
        
        # self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        # self.m_attr_length_user_list = []
        self.m_user_batch_list = []

        self.m_ref_attr_item_batch_list = []
        self.m_ref_item_batch_list = []
        
        self.m_pos_target_list = []
        self.m_pos_len_list = []

        self.m_neg_target_list = []
        self.m_neg_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        user_itemdict_attrlist_dict = self.f_split(df)

        # print("boa_user_dict", boa_user_dict)
        max_attr_len = args.max_seq_length

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        pos_attr_list = df.pos_attr.tolist()
        neg_attr_list = df.neg_attr.tolist()

        for sample_index in range(self.m_sample_num):
            userid_i = userid_list[sample_index]
            itemid_i = itemid_list[sample_index]
            pos_attrlist_i = [int(pos_attr_list[sample_index])]
            neg_attrlist_i = list(neg_attr_list[sample_index])
            neg_attrlist_i = [int(j) for j in neg_attrlist_i]

            if userid_i not in user_itemdict_attrlist_dict:
                print("user no reference in training error")
                continue

            ref_attrlist_list = []
            ref_itemid_list = []

            itemdict_attrlist_dict = user_itemdict_attrlist_dict[userid_i]
            for itemid_j in itemdict_attrlist_dict:
                if itemid_j == itemid_i:
                    continue

                attrlist_j = itemdict_attrlist_dict[itemid_j]

                ref_attrlist_list.append(attrlist_j)
                ref_itemid_list.append(itemid_j)

            self.m_ref_attr_item_batch_list.append(ref_attrlist_list)
            self.m_ref_item_batch_list.append(ref_itemid_list)

            """
            scale the item freq into the range [0, 1]
            """

            # def max_min_scale(val_list):
            #     vals = np.array(val_list)
            #     min_val = min(vals)
            #     max_val = max(vals)
            #     if max_val == min_val:
            #         scale_vals = np.zeros_like(vals)
            #         # print("scale_vals", scale_vals)
            #     else:
            #         scale_vals = (vals-min_val)/(max_val-min_val)

            #     scale_vals = scale_vals+1.0
            #     scale_val_list = list(scale_vals)
    
            #     return scale_val_list

            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_pos_target_list.append(pos_attrlist_i)
            self.m_pos_len_list.append(len(pos_attrlist_i))

            self.m_neg_target_list.append(neg_attrlist_i)
            self.m_neg_len_list.append(len(neg_attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list))
        # exit()

    ### split the data into hierarchical
    ### per user 10 items
    def f_split(self, df):
        user_item_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        user_itemdict_attrlist_dict = {}

        maxitemnum_user = 20

        for (userid_i, itemid_i) in user_item_attrlist_dict:
            attr_i = user_item_attrlist_dict[(userid_i, itemid_i)]
            
            if userid_i not in user_itemdict_attrlist_dict:
                user_itemdict_attrlist_dict[userid_i] = {}

            if itemid_i not in user_itemdict_attrlist_dict[userid_i]:
                user_itemdict_attrlist_dict[userid_i][itemid_i] = []
            user_itemdict_attrlist_dict[userid_i][itemid_i] = attr_i
            
        reserve_user_itemdict_attrlist_dict = {}
        for userid_i in user_itemdict_attrlist_dict:
            itemdict_attrlist_dict_i = user_itemdict_attrlist_dict[userid_i]
            
            itemid_list_i = list(itemdict_attrlist_dict_i.keys())
            random.shuffle(itemid_list_i)

            reserve_itemnum = len(itemid_list_i)
            if reserve_itemnum > maxitemnum_user:
                reserve_itemnum = reserve_itemnum

            if userid_i not in reserve_user_itemdict_attrlist_dict:
                reserve_user_itemdict_attrlist_dict[userid_i] = {}

            for j in range(reserve_itemnum):
                itemid_ij = itemid_list_i[j]
                attr_ij = itemdict_attrlist_dict_i[itemid_ij]
                if itemid_ij not in reserve_user_itemdict_attrlist_dict[userid_i]:
                    reserve_user_itemdict_attrlist_dict[userid_i][itemid_ij] = []
                reserve_user_itemdict_attrlist_dict[userid_i][itemid_ij] = attr_ij
    
        return reserve_user_itemdict_attrlist_dict

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        ref_attr_item_list_i = self.m_ref_attr_item_batch_list[i]

        ref_item_list_i = self.m_ref_item_batch_list[i]

        item_i = self.m_item_batch_list[i]

        user_i = self.m_user_batch_list[i]

        pos_target_i = self.m_pos_target_list[i]
        pos_len_i = self.m_pos_len_list[i]

        neg_target_i = self.m_neg_target_list[i]
        neg_len_i = self.m_neg_len_list[i]

        sample_i = {"ref_attr_item": ref_attr_item_list_i, "ref_item": ref_item_list_i,  "user": user_i,  "item": item_i, "pos_target": pos_target_i, "pos_len": pos_len_i, "neg_target": neg_target_i, "neg_len": neg_len_i, "max_seq_len": self.m_max_seq_len}

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
        
        ref_attr_len_item_iter = []
        ref_item_len_iter = []
        # ref_item_mask_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            ref_attr_item_i = sample_i["ref_attr_item"]
            for j in ref_attr_item_i:
                ref_attr_len_item_iter.append(len(j))

            ref_item_i = sample_i["ref_item"]
            ref_item_len = len(ref_item_i)
            ref_item_len_iter.append(ref_item_len)

            assert len(ref_item_i) == len(ref_attr_item_i)

            neg_len_i = sample_i["neg_len"]
            neg_len_iter.append(neg_len_i)

        max_ref_attr_len_item = max(ref_attr_len_item_iter)
        max_ref_item_len = max(ref_item_len_iter)

        # ref_item_mask_iter = [i for i, ref_item_len_i in enumerate(ref_item_len_iter) for j in range(ref_item_len_i)]
       
        # max_pos_targetlen_iter = max(pos_len_iter)
        max_neg_targetlen_iter = max(neg_len_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        ref_attr_item_iter = []
        ref_attr_mask_item_iter = []

        ref_item_iter = []
        ref_item_mask_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            ref_attr_item_i = copy.deepcopy(sample_i["ref_attr_item"])
            
            ref_item_i = copy.deepcopy(sample_i["ref_item"])

            ref_item_num_i = len(ref_attr_item_i)

            assert len(ref_item_i) == len(ref_attr_item_i)

            for j in ref_attr_item_i:
                len_j = len(j)
                j.extend([freq_pad_id]*(max_ref_attr_len_item-len_j))
                ref_attr_item_iter.append(j)

                mask_j = [1 for i in range(len_j)]+[0]*(max_ref_attr_len_item-len_j)
                ref_attr_mask_item_iter.append(mask_j)

            # ref_item_i.extend([-1]*(max_ref_item_len-ref_item_num_i))
            # ref_item_iter.append(ref_item_i)

            ref_item_mask_i = [1 for i in range(ref_item_num_i)]
            ref_item_mask_i.extend([0]*(max_ref_item_len-ref_item_num_i))
            ref_item_mask_iter.append(ref_item_mask_i)
            
            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])
            pos_target_iter.append(pos_target_i)

            pos_len_i = 1
            pos_len_iter.append(pos_len_i)
            
            neg_target_i = copy.deepcopy(sample_i["neg_target"])
            neg_len_i = sample_i["neg_len"]
            neg_target_i.extend([pad_id]*(max_neg_targetlen_iter-neg_len_i))
            neg_target_iter.append(neg_target_i)

        ref_attr_item_iter_tensor = torch.from_numpy(np.array(ref_attr_item_iter)).long()
        # ref_attr_len_item_iter_tensor = torch.from_numpy(np.array(ref_attr_len_item_iter)).long()
        ref_attr_mask_item_iter_tensor = torch.from_numpy(np.array(ref_attr_mask_item_iter)).long()

        ref_item_iter_tensor = torch.from_numpy(np.array(ref_item_iter)).long()
        ref_item_mask_iter_tensor = torch.from_numpy(np.array(ref_item_mask_iter)).long()
        # ref_item_len_iter_tensor = torch.from_numpy(np.array(ref_item_len_iter)).long()

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        pos_target_iter_tensor = torch.from_numpy(np.array(pos_target_iter)).long()
        pos_len_iter_tensor = torch.from_numpy(np.array(pos_len_iter)).long()

        neg_target_iter_tensor = torch.from_numpy(np.array(neg_target_iter)).long()
        neg_len_iter_tensor = torch.from_numpy(np.array(neg_len_iter)).long()

        return ref_attr_item_iter_tensor, ref_attr_mask_item_iter_tensor, ref_item_iter_tensor, ref_item_mask_iter_tensor, user_iter_tensor, item_iter_tensor, pos_target_iter_tensor, pos_len_iter_tensor, neg_target_iter_tensor, neg_len_iter_tensor

class _WINE_TEST(Dataset):
    def __init__(self, args, vocab_obj, train_df, df):
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
    
        self.m_item_batch_list = []
        self.m_user_batch_list = []
        
        user_itemdict_attrlist_dict = self.f_split(train_df)

        self.m_target_list = []
        self.m_target_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        
        attr_list = df.attr.tolist()

        max_attr_len = args.max_seq_length

        self.m_ref_attr_item_batch_list = []
        self.m_ref_item_batch_list = []

        for sample_index in range(self.m_sample_num):
        # for sample_index in range(20):
            userid_i = userid_list[sample_index]
            itemid_i = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])
            attrlist_i = [int(j) for j in attrlist_i]

            if userid_i not in user_itemdict_attrlist_dict:
                print("user no reference in training error")
                continue

            ref_attrlist_list = []
            ref_itemid_list = []

            # print("userid", userid_i)
            # print("itemid", itemid_i)

            itemdict_attrlist_dict = user_itemdict_attrlist_dict[userid_i]

            # print("attrlist_i", attrlist_i)
            # print("itemdict_attrlist_dict", itemdict_attrlist_dict)
            
            for itemid_j in itemdict_attrlist_dict:
                if itemid_j == itemid_i:
                    continue

                attrlist_j = itemdict_attrlist_dict[itemid_j]

                ref_attrlist_list.append(attrlist_j)
                ref_itemid_list.append(itemid_j)

            self.m_ref_attr_item_batch_list.append(ref_attrlist_list)
            self.m_ref_item_batch_list.append(ref_itemid_list)

            # """
            # scale the item freq into the range [0, 1]
            # """

            # def max_min_scale(val_list):
            #     vals = np.array(val_list)
            #     min_val = min(vals)
            #     max_val = max(vals)
            #     if max_val == min_val:
            #         scale_vals = np.zeros_like(vals)
            #     else:
            #         scale_vals = (vals-min_val)/(max_val-min_val)

            #     scale_vals = scale_vals+1.0
            #     scale_val_list = list(scale_vals)

            #     return scale_val_list

            self.m_user_batch_list.append(userid_i)
            self.m_item_batch_list.append(itemid_i)

            self.m_target_list.append(attrlist_i)
            self.m_target_len_list.append(len(attrlist_i))
        # exit()

        print("... load train data ...", len(self.m_item_batch_list))
    
    def f_split(self, df):
        user_item_attrlist_dict = dict(df.groupby(['userid', 'itemid']).pos_attr.apply(list))

        user_itemdict_attrlist_dict = {}

        maxitemnum_user = 20

        for (userid_i, itemid_i) in user_item_attrlist_dict:
            attr_i = user_item_attrlist_dict[(userid_i, itemid_i)]
            
            if userid_i not in user_itemdict_attrlist_dict:
                user_itemdict_attrlist_dict[userid_i] = {}

            if itemid_i not in user_itemdict_attrlist_dict[userid_i]:
                user_itemdict_attrlist_dict[userid_i][itemid_i] = []
            user_itemdict_attrlist_dict[userid_i][itemid_i] = attr_i
            
        reserve_user_itemdict_attrlist_dict = {}
        for userid_i in user_itemdict_attrlist_dict:
            itemdict_attrlist_dict_i = user_itemdict_attrlist_dict[userid_i]
            
            itemid_list_i = list(itemdict_attrlist_dict_i.keys())
            random.shuffle(itemid_list_i)

            reserve_itemnum = len(itemid_list_i)
            if reserve_itemnum > maxitemnum_user:
                reserve_itemnum = reserve_itemnum

            if userid_i not in reserve_user_itemdict_attrlist_dict:
                reserve_user_itemdict_attrlist_dict[userid_i] = {}

            for j in range(reserve_itemnum):
                itemid_ij = itemid_list_i[j]
                attr_ij = itemdict_attrlist_dict_i[itemid_ij]
                if itemid_ij not in reserve_user_itemdict_attrlist_dict[userid_i]:
                    reserve_user_itemdict_attrlist_dict[userid_i][itemid_ij] = []
                reserve_user_itemdict_attrlist_dict[userid_i][itemid_ij] = attr_ij
    
        return reserve_user_itemdict_attrlist_dict


    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        ref_attr_item_list_i = self.m_ref_attr_item_batch_list[i]
        ref_item_list_i = self.m_ref_item_batch_list[i]
        
        item_i = self.m_item_batch_list[i]
        user_i = self.m_user_batch_list[i]

        target_i = self.m_target_list[i]
        target_len_i = self.m_target_len_list[i]

        sample_i = {"ref_attr_item": ref_attr_item_list_i, "ref_item": ref_item_list_i,  "user": user_i,  "item": item_i,  "target": target_i, "target_len": target_len_i, "max_seq_length": self.m_max_seq_len}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        item_iter = []
        user_iter = []

        ref_attr_len_item_iter = []
        ref_item_len_iter = []

        target_iter = []
        target_len_iter = []
        target_mask_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            ref_attr_item_i = sample_i["ref_attr_item"]
            for j in ref_attr_item_i:
                ref_attr_len_item_iter.append(len(j))

            ref_item_i = sample_i["ref_item"]
            ref_item_len = len(ref_item_i)
            ref_item_len_iter.append(ref_item_len)

            assert len(ref_item_i) == len(ref_attr_item_i)

            target_len_i = sample_i["target_len"]
            target_len_iter.append(target_len_i)

        max_ref_attr_len_item = max(ref_attr_len_item_iter)
        max_ref_item_len = max(ref_item_len_iter)

        max_targetlen_iter = max(target_len_iter)

        # print("max_pos_targetlen_iter", max_pos_targetlen_iter)
        # print("max_neg_targetlen_iter", max_neg_targetlen_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0
        ind_pad_id = 0

        ref_attr_item_iter = []
        ref_attr_mask_item_iter = []

        ref_item_iter = []
        ref_item_mask_iter = []
        
        for i in range(batch_size):
            sample_i = batch[i]

            ref_attr_item_i = copy.deepcopy(sample_i["ref_attr_item"])
            ref_item_i = copy.deepcopy(sample_i["ref_item"])

            ref_item_num_i = len(ref_attr_item_i)
            assert len(ref_item_i) == len(ref_attr_item_i)

            for j in ref_attr_item_i:
                len_j = len(j)
                j.extend([freq_pad_id]*(max_ref_attr_len_item-len_j))
                ref_attr_item_iter.append(j)

                mask_j = [1 for i in range(len_j)]+[0]*(max_ref_attr_len_item-len_j)
                ref_attr_mask_item_iter.append(mask_j)

            ref_item_mask_i = [1 for i in range(ref_item_num_i)]
            ref_item_mask_i.extend([0]*(max_ref_item_len-ref_item_num_i))
            ref_item_mask_iter.append(ref_item_mask_i)

            # ref_item_i.extend([-1]*(max_ref_item_len-ref_item_num_i))
            # ref_item_iter.append(ref_item_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            target_i = copy.deepcopy(sample_i["target"])

            target_len_i = sample_i["target_len"]
            target_i.extend([-1]*(max_targetlen_iter-target_len_i))
            target_iter.append(target_i)

            target_mask_iter.append([1]*target_len_i+[0]*(max_targetlen_iter-target_len_i))
        
        ref_attr_item_iter_tensor = torch.from_numpy(np.array(ref_attr_item_iter)).long()
        
        # ref_attr_len_item_iter_tensor = torch.from_numpy(np.array(ref_attr_len_item_iter)).long()
        ref_attr_mask_item_iter_tensor = torch.from_numpy(np.array(ref_attr_mask_item_iter)).long()

        ref_item_iter_tensor = torch.from_numpy(np.array(ref_item_iter)).long()
        # ref_item_len_iter_tensor = torch.from_numpy(np.array(ref_item_len_iter)).long()
        ref_item_mask_iter_tensor = torch.from_numpy(np.array(ref_item_mask_iter)).long()

        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
       
        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
        target_mask_iter_tensor = torch.from_numpy(np.array(target_mask_iter)).long()

        return  ref_attr_item_iter_tensor, ref_attr_mask_item_iter_tensor, ref_item_iter_tensor, ref_item_mask_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor, target_mask_iter_tensor
