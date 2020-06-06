###

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
from clothing import _CLOTHING, _CLOTHING_TEST
from movie import _MOVIE, _MOVIE_TEST
from nltk import ngrams

class _Data():
    def __init__(self):
        print("data")

        self.m_word_map_user_perturb = {}
        self.m_word_map_item_perturb ={}
        self.m_word_map_local_perturb = {}

    def f_create_data(self, args):
        self.m_min_occ = args.min_occ
        self.m_max_line = 1e8

        self.m_data_dir = args.data_dir
        self.m_data_name = args.data_name
        self.m_raw_data_file = args.data_file
        self.m_raw_data_path = os.path.join(self.m_data_dir, self.m_raw_data_file)
       
        data = pd.read_pickle(self.m_raw_data_path)
        train_df = data["train"]
        valid_df = data["valid"]

        train_reviews = train_df.review
        train_item_ids = train_df.itemid
        train_user_ids = train_df.userid

        valid_reviews = valid_df.review
        valid_item_ids = valid_df.itemid
        valid_user_ids = valid_df.userid

        print("loading train reviews")

        ss_time = datetime.datetime.now()

        self.m_max_line = 1e4
        print("max line num", self.m_max_line)

        train_file = "clothing_train.txt"
        train_file = os.path.join(self.m_data_dir, train_file)
        train_f = open(train_file, "w")

        for index, review in enumerate(train_reviews):
            if index > self.m_max_line:
                break

            train_f.write(review)
            train_f.write("\n")

        train_f.close()
    
        e_time = datetime.datetime.now()
        print("load training duration", e_time-ss_time)
        
        s_time = datetime.datetime.now()
        print("loading valid reviews")

        test_file = "clothing_test.txt"
        test_file = os.path.join(self.m_data_dir, test_file)
        test_f = open(test_file, "w")

        for index, review in enumerate(valid_reviews):

            if index > self.m_max_line:
                break

            test_f.write(review)
            test_f.write("\n")  
          
        test_f.close()

### python perturb_data.py --data_dir "../data/amazon/movie" --data_file "processed_amazon_movie.pickle" --output_file "pro.pickle"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/amazon/')
    parser.add_argument('-dn', '--data_name', type=str, default='amazon')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=5)
    parser.add_argument('--data_file', type=str, default="raw_data.pickle")
    parser.add_argument('--output_file', type=str, default=".pickle")
    parser.add_argument('--user_word_file', type=str, default="user_word_score.pickle")
    parser.add_argument('--item_word_file', type=str, default="item_word_score.pickle")

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    data_obj = _Data()
    data_obj.f_create_data(args)
