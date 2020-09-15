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
from torch.utils.data import dataset 

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from movie import _MOVIE, _MOVIE_TEST

class _DATA():
    def __init__(self):
        print("data")

    def f_load_data_movie(self, args):
        self.m_data_name = args.data_name

        # train_data_file = args.data_dir+"/sampled_train_100.pickle"
        # valid_data_file = args.data_dir+"/sampled_valid.pickle"
        # test_data_file = args.data_dir+"/sampled_test.pickle"
        train_data_file = args.data_dir+"/train_100.pickle"
        valid_data_file = args.data_dir+"/valid.pickle"
        test_data_file = args.data_dir+"/valid.pickle"

        # train_data_file = args.data_dir+"/self_attn_train_100.pickle"
        # valid_data_file = args.data_dir+"/self_attn_valid.pickle"
        # test_data_file = args.data_dir+"/self_attn_valid.pickle"

        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        self.m_vocab_file = args.vocab_file

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['t2i'], vocab['i2t'])
        vocab_obj.f_set_user(vocab['u2i'])
        vocab_obj.f_set_item(vocab['i2i'])

        train_user_num = train_df.userid.nunique()
        print("train user num", train_user_num)

        train_item_num = train_df.itemid.nunique()
        print("train item num", train_item_num)

        train_pos_tag_num = train_df.pos_tagid.nunique()
        print("train tag num", train_pos_tag_num)

        train_data = _MOVIE(args,  train_df)
        valid_data = _MOVIE_TEST(args, valid_df)

        batch_size = args.batch_size
 
        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj

    def f_load_data_yelp(self, args):
        self.m_data_name = args.data_name

        train_data_file = args.data_dir+"/sampled_train_100.pickle"
        valid_data_file = args.data_dir+"/sampled_valid.pickle"
        test_data_file = args.data_dir+"/sampled_test.pickle"
        # train_data_file = args.data_dir+"/train_100.pickle"
        # valid_data_file = args.data_dir+"/valid.pickle"
        # test_data_file = args.data_dir+"/valid.pickle"

        # train_data_file = args.data_dir+"/self_attn_train_100.pickle"
        # valid_data_file = args.data_dir+"/self_attn_valid.pickle"
        # test_data_file = args.data_dir+"/self_attn_valid.pickle"

        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        self.m_vocab_file = args.vocab_file

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['t2i'], vocab['i2t'])
        vocab_obj.f_set_user(vocab['u2i'])
        vocab_obj.f_set_item(vocab['i2i'])

        train_user_num = train_df.userid.nunique()
        print("train user num", train_user_num)

        train_item_num = train_df.itemid.nunique()
        print("train item num", train_item_num)

        train_pos_tag_num = train_df.pos_tagid.nunique()
        print("train tag num", train_pos_tag_num)

        train_data = _MOVIE(args,  train_df)
        valid_data = _MOVIE_TEST(args, valid_df)

        batch_size = args.batch_size
 
        if args.parallel:
            train_sampler = DistributedSampler(dataset=train_data)
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, collate_fn=train_data.collate)
        else:
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_data.collate)
        test_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=valid_data.collate)

        return train_loader, test_loader, vocab_obj

class _Vocab():
    def __init__(self):
        self.m_w2i = None
        self.m_i2w = None
        self.m_vocab_size = 0
        self.m_user2uid = None
        self.m_item2iid = None

        self.m_user_num = 0
        self.m_item_num = 0
    
    def f_set_vocab(self, w2i, i2w):
        self.m_w2i = w2i
        self.m_i2w = i2w
        self.m_vocab_size = self.vocab_size

    def f_set_user(self, user2uid):
        self.m_user2uid = user2uid
        self.m_user_num = len(self.m_user2uid)

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
