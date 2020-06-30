import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import random
import pandas as pd
import argparse
import copy
from nltk.corpus import stopwords
from utils import OrderedCounter
from review import _Review
from item import _Item
from user import _User
import pickle
import string
import datetime
from collections import Counter 
from scipy import sparse

class _YELP(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_min_occ = args.min_occ
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        ###get length
        
        length_list = []
        self.m_length_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_input_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_target_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_user_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_item_batch_list = [[] for i in range(self.m_batch_num)]

        self.m_user2uid = {}
        self.m_item2iid = {}
       
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        review_list = df.review.tolist()
        tokens_list = df.token_idxs.tolist()

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            review = review_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            length_list.append(len_review)

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)
            
            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            if batch_index >= self.m_batch_num:
                break
            
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = tokens[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])

            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)

            # uid = self.m_user2uid[user_id]
            self.m_user_batch_list[batch_index].append(user_id)

            # iid = self.m_item2iid[item_id]
            self.m_item_batch_list[batch_index].append(item_id)

    def __iter__(self):
        print("shuffling")  
        
        batch_index_list = np.random.permutation(self.m_batch_num)

        for batch_i in range(self.m_batch_num):
            batch_index = batch_index_list[batch_i]

            length_batch = self.m_length_batch_list[batch_index]
            input_batch = self.m_input_batch_list[batch_index]
            target_batch = self.m_target_batch_list[batch_index]

            user_batch = self.m_user_batch_list[batch_index]
            item_batch = self.m_item_batch_list[batch_index]

            input_batch_iter = []
            target_batch_iter = []
           
            max_length_batch = max(length_batch)

            for sent_i, _ in enumerate(input_batch):

                length_i = length_batch[sent_i]
                
                input_i_iter = copy.deepcopy(input_batch[sent_i])
                target_i_iter = copy.deepcopy(target_batch[sent_i])

                input_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))
                target_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))

                input_batch_iter.append(input_i_iter)

                target_batch_iter.append(target_i_iter)

            length_batch_tensor = torch.from_numpy(np.array(length_batch)).long()
            input_batch_iter_tensor = torch.from_numpy(np.array(input_batch_iter)).long()

            target_batch_iter_tensor = torch.from_numpy(np.array(target_batch_iter)).long()

            user_iter = user_batch
            item_iter = item_batch

            user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
            item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

            yield input_batch_iter_tensor, target_batch_iter_tensor, length_batch_tensor, user_iter_tensor, item_iter_tensor

class _YELP_TEST(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_min_occ = args.min_occ
        self.m_batch_size = args.batch_size
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        ###get length
        
        length_list = []
        self.m_length_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_input_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_target_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_user_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_item_batch_list = [[] for i in range(self.m_batch_num)]

        self.m_user2uid = {}
        self.m_item2iid = {}
       
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        review_list = df.review.tolist()
        tokens_list = df.token_idxs.tolist()

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            review = review_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            length_list.append(len_review)

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)
            
            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            if batch_index >= self.m_batch_num:
                break
            
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = tokens[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])

            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)

            uid = self.m_user2uid[user_id]
            self.m_user_batch_list[batch_index].append(uid)

            iid = self.m_item2iid[item_id]
            self.m_item_batch_list[batch_index].append(iid)

    def __iter__(self):
        print("shuffling")  
        
        batch_index_list = np.random.permutation(self.m_batch_num)

        for batch_i in range(self.m_batch_num):
            batch_index = batch_index_list[batch_i]

            length_batch = self.m_length_batch_list[batch_index]
            input_batch = self.m_input_batch_list[batch_index]
            target_batch = self.m_target_batch_list[batch_index]

            user_batch = self.m_user_batch_list[batch_index]
            item_batch = self.m_item_batch_list[batch_index]

            input_batch_iter = []
            target_batch_iter = []
           
            max_length_batch = max(length_batch)

            for sent_i, _ in enumerate(input_batch):

                length_i = length_batch[sent_i]
                
                input_i_iter = copy.deepcopy(input_batch[sent_i])
                target_i_iter = copy.deepcopy(target_batch[sent_i])

                input_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))
                target_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))

                input_batch_iter.append(input_i_iter)

                target_batch_iter.append(target_i_iter)

            length_batch_tensor = torch.from_numpy(np.array(length_batch)).long()
            input_batch_iter_tensor = torch.from_numpy(np.array(input_batch_iter)).long()

            target_batch_iter_tensor = torch.from_numpy(np.array(target_batch_iter)).long()

            user_iter = user_batch
            item_iter = item_batch
            
            user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
            item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

            yield input_batch_iter_tensor, target_batch_iter_tensor, length_batch_tensor, user_iter_tensor, item_iter_tensor

