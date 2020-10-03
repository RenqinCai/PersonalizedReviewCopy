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
import pickle
import string
import datetime
from collections import Counter 
# from cloth import _CLOTH, _CLOTH_TEST
# from clothing import _CLOTHING, _CLOTHING_TEST
# from movie import _MOVIE, _MOVIE_TEST
# from yelp_edu import _YELP, _YELP_TEST
from yelp_restaurant import _YELP_RESTAURANT
from movie import _MOVIE, _MOVIE_TEST
from wine import _WINE, _WINE_TEST
from ratebeer import _RATEBEER, _RATEBEER_TEST

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class _DATA():
    def __init__(self):
        print("data")

    def f_load_data_yelp_restaurant(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file
        self.m_item_boa_file = args.item_boa_file
        self.m_user_boa_file = args.user_boa_file

        train_data_file = args.data_dir+"/new_train.pickle"
        valid_data_file = args.data_dir+"/new_valid.pickle"
        test_data_file = args.data_dir+"/new_valid.pickle"

        # train_data_file = args.data_dir+"/train.pickle"
        # valid_data_file = args.data_dir+"/valid.pickle"
        # test_data_file = args.data_dir+"/test.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()
        print("user num", user_num)

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_item_boa_file), 'r',encoding='utf8') as f:
            item_boa_dict = json.loads(f.read())

        # with open(os.path.join(args.data_dir, self.m_user_boa_file), 'r', encoding='utf8') as f:
        #     user_boa_dict = json.loads(f.read())

        user_boa_dict = {}

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['a2i'], vocab['i2a'])
        vocab_obj.f_set_user_num(user_num)

        global_user2iid = vocab['user_index']
        global_item2iid = vocab['item_index']

        vocab_obj.f_set_user(global_user2iid)
        vocab_obj.f_set_item(global_item2iid)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _YELP_RESTAURANT(args, vocab_obj, train_df, item_boa_dict, user_boa_dict)
        valid_data = _YELP_RESTAURANT(args, vocab_obj, valid_df, item_boa_dict, user_boa_dict)

        batch_size = args.batch_size

        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)

            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj

    def f_load_data_movie(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file
        self.m_item_boa_file = args.item_boa_file
        self.m_user_boa_file = args.user_boa_file

        train_data_file = args.data_dir+"/new_train.pickle"
        valid_data_file = args.data_dir+"/new_valid.pickle"
        test_data_file = args.data_dir+"/new_valid.pickle"

        # train_data_file = args.data_dir+"/train.pickle"
        # valid_data_file = args.data_dir+"/valid.pickle"
        # test_data_file = args.data_dir+"/test.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()
        print("user num", user_num)

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_item_boa_file), 'r',encoding='utf8') as f:
            item_boa_dict = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_user_boa_file), 'r', encoding='utf8') as f:
            user_boa_dict = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['t2i'], vocab['i2t'])
        vocab_obj.f_set_user_num(user_num)

        global_user2iid = vocab['user_index']
        global_item2iid = vocab['item_index']

        vocab_obj.f_set_user(global_user2iid)
        vocab_obj.f_set_item(global_item2iid)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _MOVIE(args, vocab_obj, train_df, item_boa_dict, user_boa_dict)
        valid_data = _MOVIE_TEST(args, vocab_obj, valid_df, item_boa_dict, user_boa_dict)

        batch_size = args.batch_size

        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)

            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=8, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj   

    def f_load_data_wine(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file
        self.m_item_boa_file = args.item_boa_file
        self.m_user_boa_file = args.user_boa_file

        # train_data_file = args.data_dir+"/new_train.pickle"
        # valid_data_file = args.data_dir+"/new_valid.pickle"
        # test_data_file = args.data_dir+"/new_valid.pickle"

        train_data_file = args.data_dir+"/train.pickle"
        valid_data_file = args.data_dir+"/valid.pickle"
        test_data_file = args.data_dir+"/valid.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()
        print("user num", user_num)

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_item_boa_file), 'r',encoding='utf8') as f:
            item_boa_dict = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_user_boa_file), 'r', encoding='utf8') as f:
            user_boa_dict = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['t2i'], vocab['i2t'])
        vocab_obj.f_set_user_num(user_num)

        global_user2iid = vocab['user_index']
        global_item2iid = vocab['item_index']

        vocab_obj.f_set_user(global_user2iid)
        vocab_obj.f_set_item(global_item2iid)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _WINE(args, vocab_obj, train_df, item_boa_dict, user_boa_dict)
        valid_data = _WINE_TEST(args, vocab_obj, valid_df, item_boa_dict, user_boa_dict)

        batch_size = args.batch_size

        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)

            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=8, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj   

    def f_load_data_beer(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file
        self.m_item_boa_file = args.item_boa_file
        self.m_user_boa_file = args.user_boa_file

        # train_data_file = args.data_dir+"/new_train.pickle"
        # valid_data_file = args.data_dir+"/new_valid.pickle"
        # test_data_file = args.data_dir+"/new_valid.pickle"

        train_data_file = args.data_dir+"/train_debug.pickle"
        valid_data_file = args.data_dir+"/valid.pickle"
        test_data_file = args.data_dir+"/valid.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()
        print("user num", user_num)

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_item_boa_file), 'r',encoding='utf8') as f:
            item_boa_dict = json.loads(f.read())

        with open(os.path.join(args.data_dir, self.m_user_boa_file), 'r', encoding='utf8') as f:
            user_boa_dict = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['t2i'], vocab['i2t'])
        vocab_obj.f_set_user_num(user_num)

        global_user2iid = vocab['user_index']
        global_item2iid = vocab['item_index']

        # global_user2iid = vocab['u2i']
        # global_item2iid = vocab['i2i']

        vocab_obj.f_set_user(global_user2iid)
        vocab_obj.f_set_item(global_item2iid)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _RATEBEER(args, vocab_obj, train_df, item_boa_dict, user_boa_dict)
        valid_data = _RATEBEER_TEST(args, vocab_obj, valid_df, item_boa_dict, user_boa_dict)

        batch_size = args.batch_size

        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)

            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj   

class _Vocab():
    def __init__(self):
        self.m_w2i = None
        self.m_i2w = None
        self.m_vocab_size = 0
        self.m_user2uid = None
        self.m_user_size = 0
    
    def f_set_vocab(self, w2i, i2w):
        self.m_w2i = w2i
        self.m_i2w = i2w
        self.m_vocab_size = self.vocab_size

    def f_set_user(self, user2uid):
        self.m_user2uid = user2uid
        self.m_user_num = len(self.m_user2uid)

    def f_set_user_num(self, user_num):
        self.m_user_num = user_num

    def f_set_item(self, item2iid):
        self.m_item2iid = item2iid
        self.m_item_num = len(item2iid)

    @property
    def item_num(self):
        return self.m_item_num

    @property
    def user_num(self):
        return self.m_user_num

    @property
    def vocab_size(self):
        return len(self.m_w2i)

    @property
    def pad_idx(self):
        return 0
        # return self.m_w2i['<pad>']

    @property
    def unk_idx(self):
        return 1
        # return self.m_w2i['<unk>']

    @property
    def sos_idx(self):
        return 2
        # return self.m_w2i['<sos>']

    @property
    def eos_idx(self):
        return 3
        # return self.m_w2i['<eos>']



### python data.py --data_dir "../data/amazon/clothing" --data_file "processed_amazon_clothing_shoes_jewelry.pickle" --output_file "pro_v2.pickle"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/amazon/')
    parser.add_argument('-dn', '--data_name', type=str, default='amazon')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=5)
    parser.add_argument('--data_file', type=str, default="raw_data.pickle")
    parser.add_argument('--output_file', type=str, default=".pickle")

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    data_obj = _Data()
    data_obj.f_create_data(args)
