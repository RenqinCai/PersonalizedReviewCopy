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

class _MOVIE(Dataset):
    def __init__(self, args, vocab_obj, review_corpus):
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
        self.m_vocab = vocab_obj
    
        self.m_sample_num = len(review_corpus)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
            self.m_batch_num += 1

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

        for sample_index in range(self.m_sample_num):
            review_obj = review_corpus[sample_index]

            user_id = review_obj.m_user_id
            item_id = review_obj.m_item_id

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)

            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            length_list.append(len_review)

        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            residual_index = i-batch_index*self.m_batch_size
            
            review_obj = review_corpus[sample_index]
            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = [self.m_sos_id] + word_ids_review[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])

            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)

            user_p_review = review_obj.m_user_perturb_words
            target_user_p_review = [self.m_sos_id] + user_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_user_p_target_batch_list[batch_index].append(target_user_p_review)

            item_p_review = review_obj.m_item_perturb_words
            target_item_p_review = [self.m_sos_id] + item_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_item_p_target_batch_list[batch_index].append(target_item_p_review)

            local_p_review = review_obj.m_local_perturb_words
            target_local_p_review = [self.m_sos_id] + local_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_local_p_target_batch_list[batch_index].append(target_local_p_review)

            user_id = int(review_obj.m_user_id)
            self.m_user_batch_list[batch_index].append(user_id)

            item_id = int(review_obj.m_item_id)
            self.m_item_batch_list[batch_index].append(item_id)

        print("loaded data")

    def __iter__(self):
        print("shuffling")
        
        batch_index_list = np.random.permutation(self.m_batch_num)
        
        """
        random_flag
        0: "none": user+item+local z+s+l
        1: "local": user+item z+s new_local
        2: "user": item+local s+l new_user
        3: "item": user+local z+l new_item
        """ 
        # flag_list = [0, 1, 3]

        for batch_i in range(self.m_batch_num):
            batch_index = batch_index_list[batch_i]
            # s_time = datetime.datetime.now()

            # random_flag = random.randint(0, 2)
            # random_flag = random.randint(0, 3)
            # random_flag = random.choice(flag_list)
            random_flag = 3
 
            input_batch = self.m_input_batch_list[batch_index]
            input_length_batch = self.m_length_batch_list[batch_index]

            user_batch = self.m_user_batch_list[batch_index]
            item_batch = self.m_item_batch_list[batch_index]

            target_batch = None
            if random_flag == 0:
                target_batch = self.m_target_batch_list[batch_index]
            elif random_flag == 1:
                target_batch = self.m_local_p_target_batch_list[batch_index]
            elif random_flag == 2:
                target_batch = self.m_user_p_target_batch_list[batch_index]
            elif random_flag == 3:
                target_batch = self.m_item_p_target_batch_list[batch_index]
            
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
                
            user_iter = user_batch
            item_iter = item_batch

            input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
            input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()

            user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
            item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

            target_length_iter_tensor = torch.from_numpy(np.array(target_length_iter)).long()
            target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
            
            yield input_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor, target_length_iter_tensor, random_flag

class _MOVIE_TEST(Dataset):
    def __init__(self, args, vocab_obj, review_corpus):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_min_occ = args.min_occ
        self.m_batch_size = args.batch_size
    
        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_item_num = vocab_obj.m_item_size
        self.m_user_num = vocab_obj.m_user_size

        self.m_sample_num = len(review_corpus)
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

        for sample_index in range(self.m_sample_num):
            review_obj = review_corpus[sample_index]

            user_id = review_obj.m_user_id
            item_id = review_obj.m_item_id

            if user_id not in self.m_user2uid:
                self.m_user2uid[user_id] = len(self.m_user2uid)
            
            if item_id not in self.m_item2iid:
                self.m_item2iid[item_id] = len(self.m_item2iid)

            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            length_list.append(len_review)

        # sorted_length_list = sorted(length_list, reverse=True)
        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            residual_index = i-batch_index*self.m_batch_size
            if batch_index >= self.m_batch_num:
                break
            
            review_obj = review_corpus[sample_index]
            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = [self.m_sos_id] + word_ids_review[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])

            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)

            user_p_review = review_obj.m_user_perturb_words
            target_user_p_review = [self.m_sos_id] + user_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_user_p_target_batch_list[batch_index].append(target_user_p_review)

            item_p_review = review_obj.m_item_perturb_words
            target_item_p_review = [self.m_sos_id] + item_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_item_p_target_batch_list[batch_index].append(target_item_p_review)

            local_p_review = review_obj.m_local_perturb_words
            target_local_p_review = [self.m_sos_id] + local_p_review[:self.m_max_seq_len]+[self.m_eos_id]
            self.m_local_p_target_batch_list[batch_index].append(target_local_p_review)

            user_id = int(review_obj.m_user_id)
            uid = self.m_user2uid[user_id]
            self.m_user_batch_list[batch_index].append(uid)

            item_id = int(review_obj.m_item_id)
            iid = self.m_item2iid[item_id]
            self.m_item_batch_list[batch_index].append(iid)
        
        print("load valid data", len(self.m_item_batch_list), len(self.m_user_batch_list), len(self.m_input_batch_list), len(self.m_target_batch_list))

        print("loaded data")

    def __iter__(self):
        print("shuffling")
        
        batch_index_list = np.random.permutation(self.m_batch_num)
        
        random_flag = 0

        for batch_i in range(self.m_batch_num):
            batch_index = batch_index_list[batch_i]
            # s_time = datetime.datetime.now()

            # random_flag = random.randint(0, 2)
            # random_flag = random.randint(0, 3)
            # random_flag = random.choice(flag_list)
            random_flag = 3
 
            input_batch = self.m_input_batch_list[batch_index]
            input_length_batch = self.m_length_batch_list[batch_index]

            user_batch = self.m_user_batch_list[batch_index]
            item_batch = self.m_item_batch_list[batch_index]

            target_batch = None
            if random_flag == 0:
                target_batch = self.m_target_batch_list[batch_index]
            elif random_flag == 1:
                target_batch = self.m_local_p_target_batch_list[batch_index]
            elif random_flag == 2:
                target_batch = self.m_user_p_target_batch_list[batch_index]
            elif random_flag == 3:
                target_batch = self.m_item_p_target_batch_list[batch_index]
            
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
                
            user_iter = user_batch
            item_iter = item_batch

            input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
            input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()

            user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
            item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

            target_length_iter_tensor = torch.from_numpy(np.array(target_length_iter)).long()
            target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
            
            yield input_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor, target_length_iter_tensor, random_flag