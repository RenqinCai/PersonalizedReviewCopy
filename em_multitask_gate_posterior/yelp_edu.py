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
        self.m_random_flag = args.random_flag
    
        # self.m_vocab_file = "amazon_vocab.json"
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_vocab = vocab_obj

        self.m_cont_vocab_size = vocab_obj.m_cont_vocab_size
        self.m_func_vocab_size = vocab_obj.m_func_vocab_size
        self.m_stop_word_ids = vocab_obj.m_stop_word_ids

        self.m_i2w = vocab_obj.m_i2w

        self.m_cont_vocab = vocab_obj.m_cont_w2i
        self.m_func_vocab = vocab_obj.m_func_w2i
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

        ###get length
        
        length_list = []
        
        self.m_input_batch_list = []
        self.m_input_bow_batch_list = []
        self.m_input_length_batch_list = []
        self.m_user_batch_list = []
        self.m_item_batch_list = []
        self.m_target_batch_list = []
        self.m_target_l_batch_list = []
        self.m_target_length_batch_list = []
        
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

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)

            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

            input_review = tokens[:self.m_max_seq_len]
            len_review = len(input_review)

            length_list.append(len_review)

        for sample_index in range(self.m_sample_num):
            
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            review = review_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = [self.m_sos_id] + tokens[:self.m_max_seq_len]+[self.m_eos_id]

            target_l_review = [0 if target_review[i] in self.m_stop_word_ids else 1 for i in range(len(target_review))]
            new_target_reivew = []
            for i in range(len(target_review)):
                if target_l_review[i]:
                    new_target_reivew.append(self.m_cont_vocab[self.m_i2w[str(target_review[i])]])
                else:
                    new_target_reivew.append(self.m_func_vocab[self.m_i2w[str(target_review[i])]])

            target_review = new_target_reivew
            # target_review = [self.m_cont_vocab[self.m_i2w[target_review[i]]] if target_l_review[i] else self.m_func_vocab[self.m_i2w[target_review[i]]] for i in range(len(target_review))]

            input_bow = np.zeros(self.m_cont_vocab_size)
            for i in range(len(target_review)):
                if target_l_review[i]:
                    input_bow[target_review[i]] = 1

            self.m_input_bow_batch_list.append(input_bow)

            self.m_input_batch_list.append(input_review)
            self.m_input_length_batch_list.append(length_list[sample_index]+1)

            self.m_target_batch_list.append(target_review)
            self.m_target_l_batch_list.append(target_l_review)

            self.m_target_length_batch_list.append(length_list[sample_index]+2)
            
            # uid = self.m_user2uid[user_id]
            self.m_user_batch_list.append(user_id)

            # iid = self.m_item2iid[item_id]
            self.m_item_batch_list.append(item_id)
        
            # exit()
        print("... load train data ...", len(self.m_item_batch_list), len(self.m_user_batch_list), len(self.m_input_batch_list), len(self.m_target_batch_list))

    def __len__(self):
        return len(self.m_input_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx

        input_i = self.m_input_batch_list[i]
        input_bow_i = self.m_input_bow_batch_list[i]
        input_length_i = self.m_input_length_batch_list[i]
        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]
        target_i = self.m_target_batch_list[i]
        target_l_i = self.m_target_l_batch_list[i]
        target_length_i = self.m_target_length_batch_list[i]
        
        return input_i, input_bow_i, input_length_i, user_i, item_i, target_i, target_l_i, target_length_i, self.m_pad_id
            ### 0,        1,             2,             3,       4,      5,          6,          7,               8

    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        input_iter = []
        input_bow_iter = []
        input_length_iter = []
        user_iter = []
        item_iter = []
        target_iter = []
        target_l_iter = []
        target_length_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            input_length_i = sample_i[2]
            input_length_iter.append(input_length_i)

            target_length_i = sample_i[7]
            target_length_iter.append(target_length_i)

        max_input_length_iter = max(input_length_iter)
        max_target_length_iter = max(target_length_iter)

        user_iter = []
        item_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            input_i = copy.deepcopy(sample_i[0])
            input_bow_i = copy.deepcopy(sample_i[1])

            input_length_i = sample_i[2]

            pad_id = sample_i[8]

            input_i.extend([pad_id]*(max_input_length_iter-input_length_i))
            input_iter.append(input_i)

            input_bow_iter.append(input_bow_i)

            user_i = sample_i[3]
            user_iter.append(user_i)

            item_i = sample_i[4]
            item_iter.append(item_i)

            target_i = copy.deepcopy(sample_i[5])
            target_length_i = sample_i[7]

            target_l_i = copy.deepcopy(sample_i[6])

            target_l_i.extend([0]*(max_target_length_iter-target_length_i))
            target_l_iter.append(target_l_i)

            target_i.extend([pad_id]*(max_target_length_iter-target_length_i))
            target_iter.append(target_i)

        input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()
        input_bow_iter_tensor = torch.from_numpy(np.array(input_bow_iter)).float()

        input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
        
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        target_l_iter_tensor = torch.from_numpy(np.array(target_l_iter)).long()

        target_length_iter_tensor = torch.from_numpy(np.array(target_length_iter)).long()
        
        random_flag = 0

        # print("random flag", random_flag)

        return input_iter_tensor, input_bow_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor, target_l_iter_tensor, target_length_iter_tensor, random_flag

class _YELP_TEST(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_min_occ = args.min_occ
        self.m_batch_size = args.batch_size
        self.m_random_flag = args.random_flag
    
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
        self.m_user_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_item_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_target_batch_list = [[] for i in range(self.m_batch_num)]
        
        self.m_user_p_target_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_item_p_target_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_local_p_target_batch_list = [[] for i in range(self.m_batch_num)]
        
        self.m_user2uid = {}
        self.m_item2iid = {}

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        review_list = df.review.tolist()
        tokens_list = df.token_idxs.tolist()

        total_word_num = 0
        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            review = review_list[sample_index]
            tokens = tokens_list[sample_index]

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)
            
            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

            input_review = tokens[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            total_word_num += len(input_review)

            length_list.append(len_review)

        print("total_word_num", total_word_num)
        # exit()
        # sorted_length_list = sorted(length_list, reverse=True)
        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            residual_index = i-batch_index*self.m_batch_size
            if batch_index >= self.m_batch_num:
                break
            
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            review = review_list[sample_index]
            tokens = tokens_list[sample_index]

            input_review = tokens[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = [self.m_sos_id] + tokens[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])
            
            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)
            
            uid = self.m_user2uid[user_id]
            self.m_user_batch_list[batch_index].append(uid)

            iid = self.m_item2iid[item_id]
            self.m_item_batch_list[batch_index].append(iid)
        
            # exit()
        print("load valid data", len(self.m_item_batch_list), len(self.m_user_batch_list), len(self.m_input_batch_list), len(self.m_target_batch_list))

        print("loaded data")

    def __iter__(self):
        print("shuffling")
        
        batch_index_list = np.random.permutation(self.m_batch_num)
        
        for batch_i in range(self.m_batch_num):
            batch_index = batch_index_list[batch_i]
            # s_time = datetime.datetime.now()

            random_flag = self.m_random_flag
            if random_flag == 4:
                random_flag = random.randint(0, 3)
 
            input_batch = self.m_input_batch_list[batch_index]
            input_length_batch = self.m_length_batch_list[batch_index]

            user_batch = self.m_user_batch_list[batch_index]
            item_batch = self.m_item_batch_list[batch_index]

            target_batch = None
            if random_flag == 0:
                target_batch = self.m_target_batch_list[batch_index]

            input_iter = []
            input_length_iter = input_length_batch
            user_iter = []
            item_iter = []
            target_iter = []
            target_length_iter = []

            for target_i in target_batch:
                target_length_iter.append(len(target_i))

            max_input_length_iter = max(input_length_iter)
            max_target_length_iter = max(target_length_iter)

            for sent_i, _ in enumerate(input_batch):

                input_length_i = input_length_iter[sent_i]
                input_i_iter = copy.deepcopy(input_batch[sent_i])

                input_i_iter.extend([self.m_pad_id]*(max_input_length_iter-input_length_i))
                input_iter.append(input_i_iter)

                target_length_i = target_length_iter[sent_i]
                target_i_iter = copy.deepcopy(target_batch[sent_i])

                target_i_iter.extend([self.m_pad_id]*(max_target_length_iter-target_length_i))
                target_iter.append(target_i_iter)

                # e_time = datetime.datetime.now()
                # print("yield batch data duration", e_time-ss_time)

            # print("random flag", random_flag)

            user_iter = user_batch
            item_iter = item_batch

            # print(sum(RRe_batch_iter[0]))
            # ts_time = datetime.datetime.now()
            input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
            input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()

            user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
            item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

            target_length_iter_tensor = torch.from_numpy(np.array(target_length_iter)).long()
            target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
            
            yield input_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor, target_length_iter_tensor, random_flag