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
from typing import Dict
from utils import OrderedCounter
from review import _Review
from item import _Item
from user import _User
import pickle
import string
import datetime
from collections import Counter 
from scipy import sparse

class _YELP_RESTAURANT(Dataset):
    def __init__(self, args, vocab_obj, df, item_boa_dict):
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
        
        self.m_input_batch_list = []
        self.m_input_length_batch_list = []
        self.m_user_batch_list = []
        self.m_item_batch_list = []
        self.m_target_batch_list = []
        self.m_target_length_batch_list = []
        
        self.m_user2uid = {}
        self.m_item2iid = {}

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        boa_list = df.boa.tolist()

        for sample_index in range(self.m_sample_num):
        # for sample_index in range(1000):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            boa = boa_list[sample_index]            
            item_boa = item_boa_dict[str(item_id)]

            input_boa = item_boa
            target_boa = boa

            input_len = len(item_boa)
            target_len = len(boa)

            if not target_boa:
                # print("error", target_boa)
                continue
            
            if not input_boa:
                # print("error input boa")
                continue

            if len(input_boa) == 0:
                continue

            if len(target_boa) == 0:
                # print("empty target boa")
                continue

            self.m_input_batch_list.append(input_boa)
            self.m_target_batch_list.append(target_boa)
            
            # uid = self.m_user2uid[user_id]
            self.m_user_batch_list.append(user_id)

            # iid = self.m_item2iid[item_id]
            self.m_item_batch_list.append(item_id)

            self.m_input_length_batch_list.append(input_len)
            self.m_target_length_batch_list.append(target_len)
        
            # exit()
        print("... load train data ...", len(self.m_item_batch_list), len(self.m_user_batch_list), len(self.m_input_batch_list), len(self.m_target_batch_list), len(self.m_input_length_batch_list), len(self.m_target_length_batch_list))

    def __len__(self):
        return len(self.m_input_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx

        input_i = self.m_input_batch_list[i]
        input_length_i = self.m_input_length_batch_list[i]

        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]

        target_i = self.m_target_batch_list[i]
        target_length_i = self.m_target_length_batch_list[i]
        
        return input_i, input_length_i, user_i, item_i, target_i, target_length_i, self.m_pad_id, self.m_vocab_size
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        input_iter = []
        input_length_iter = []
        user_iter = []
        item_iter = []
        target_iter = []
        target_length_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            input_length_i = sample_i[1]
            input_length_iter.append(input_length_i)

            target_length_i = sample_i[5]
            target_length_iter.append(target_length_i)

        max_input_length_iter = max(input_length_iter)
        max_target_length_iter = max(target_length_iter)

        user_iter = []
        item_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            input_i = copy.deepcopy(sample_i[0])
            input_length_i = sample_i[1]

            # if input_i is None:
            #     print("error input is none", sample_i[0])
            # print(input_i)
            # print(len(input_i))

            pad_id = sample_i[6]
            vocab_size = sample_i[7]

            input_i.extend([pad_id]*(max_input_length_iter-input_length_i))
            input_iter.append(input_i)

            user_i = sample_i[2]
            user_iter.append(user_i)

            item_i = sample_i[3]
            item_iter.append(item_i)

            # target_i = copy.deepcopy(sample_i[4])
            # target_length_i = sample_i[5]

            # target_i.extend([pad_id]*(max_target_length_iter-target_length_i))
            # target_iter.append(target_i)
            target_index_i = copy.deepcopy(sample_i[4])
            target_i = np.zeros(vocab_size)
            target_i[np.array(target_index_i, int)] = 1
            target_i = target_i[input_i]
            target_iter.append(target_i)
        # exit()
        # print("input_iter", input_iter)
        input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()
        input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
        
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        return input_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor


def f_merge_attr(args):
    vocab_file = args.vocab_file
    data_dir = args.data_dir

    attr_file = args.attr_file

    vocab_abs_file = os.path.join(data_dir, vocab_file)
    print("vocab file", vocab_abs_file)

    with open(vocab_abs_file, 'r', encoding='utf8') as f:
        vocab = json.loads(f.read())

    w2i = vocab['w2i']
    i2w = vocab['i2w']

    train_review_num = 315594
    df_max_threshold = 0.05
    df_max_threshold = int(df_max_threshold*train_review_num)

    df_min_threshold = 20

    attr_abs_file = os.path.join(data_dir, attr_file)
    print("attr file", attr_abs_file)
    f = open(attr_abs_file, "r")

    attr_list = []
    for line in f:
        attr_i = line.strip().split(",")
        # print(attr_i)
        if len(attr_i) > 2:
            continue
        attr_name_i = attr_i[0]
        if len(attr_i) > 1:
            attr_cnt_i = int(attr_i[1])

            if df_min_threshold > attr_cnt_i:
                continue
                
            if df_max_threshold < attr_cnt_i:
                continue
            # continue
            attr_list.append(attr_name_i)
        else:
            attr_list.append(attr_name_i)
    f.close()
    
    extra_words=['<pad>','<unk>','<sos>','<eos>']

    a2i = {}
    i2a = {}

    for word in extra_words:
        aid = len(a2i)
        a2i[word] = aid
        i2a[aid] = word

    wid2aid = {}
    for w in w2i:
        if w in attr_list:
            wid = w2i[w]
            aid = len(a2i)
            a2i[w] = aid
            i2a[aid] = w
            
            wid = int(wid)
            wid2aid[wid] = aid

    print("attr num", len(a2i))

    vocab['a2i'] = a2i
    vocab['i2a'] = i2a
    vocab['wid2aid'] = wid2aid
    print(vocab['wid2aid'])

    print("save vocab to json file", vocab_file)
    with open(vocab_abs_file, 'w') as f:
        f.write(json.dumps(vocab))

def f_get_bow_item(args):
    vocab_file = args.vocab_file
    data_dir = args.data_dir

    train_data_file = data_dir+'/train.pickle'
    valid_data_file = data_dir+'/valid.pickle'
    test_data_file = data_dir+'/test.pickle'

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)

    print('train num', len(train_df))
    print('valid num', len(valid_df))
    print('test num', len(test_df))

    # exit()
    vocab_abs_file = os.path.join(args.data_dir, vocab_file)
    with open(vocab_abs_file, 'r', encoding='utf8') as f:
        vocab = json.loads(f.read())

    wid2aid = vocab['wid2aid']
    # print("wid2aid", len(wid2aid))
    # print(wid2aid)
    # print("=="*10)
    def get_review_boa(tokens):
        bow = []
        bow = [str(token) for token in tokens if str(token) in wid2aid]
        boa = [wid2aid[w] for w in bow]
        return boa
    
    train_df['boa'] = train_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    # del train_df['review']
    train_df.to_pickle(train_data_file)

    valid_df['boa'] = valid_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    # del valid_df['review']
    valid_df.to_pickle(valid_data_file)

    test_df['boa'] = test_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    # del valid_df['review']
    test_df.to_pickle(test_data_file)

    # print(train_df.iloc[0])
    # exit()
    def get_most_freq_val(val_list):
        
        a = []
        for i in val_list:
            a.extend(i)

        a = Counter(a)
        a = dict(a)
        a = [k for k, v in sorted(a.items(), key=lambda item: item[1], reverse=True)]
        
        # if len(a) == 0:
        #     print("error")

        top_k = 100
        top_a = a[:top_k]
    
        return top_a

    # item_bow_dict = dict(train_df.groupby('itemid')['boa'].apply(lambda row: get_most_freq_val(row)))

    item_boa_dict = train_df.groupby('itemid')['boa'].apply(list)
    
    item_boa_dict = item_boa_dict.apply(lambda row: get_most_freq_val(row))
    item_boa_dict = dict(item_boa_dict)

    item_boa_file_name = "item_boa.json"

    item_boa_abs_file_name = os.path.join(data_dir, item_boa_file_name)
    with open(item_boa_abs_file_name, 'w') as f:
        f.write(json.dumps(item_boa_dict))

    # item_review_list = train_df.groupby('itemid')['review'].apply(list)
    # item_review_list_dict = dict(item_review_list)
    # for item_id, item_review_list in item_review_list_dict.items():

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/yelp_restaurant")
    parser.add_argument('--vocab_file', type=str, default="vocab.json")
    parser.add_argument('--attr_file', type=str, default="attr.csv")

    args = parser.parse_args()

    f_merge_attr(args)
    f_get_bow_item(args)

