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

def f_get_cooccurrence(args):
    vocab_file = args.vocab_file
    data_dir = args.data_dir

    print("reading data")

    train_data_file = data_dir+'/new_train_DF.pickle'
    valid_data_file = data_dir+'/new_valid_DF.pickle'
    test_data_file = data_dir+'/new_test_DF.pickle'

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)

    print("columns", train_df.columns)

    print('train num', len(train_df))
    print('valid num', len(valid_df))
    print('test num', len(test_df))

    # exit()
    vocab_abs_file = os.path.join(args.data_dir, vocab_file)
    with open(vocab_abs_file, 'r', encoding='utf8') as f:
        vocab = json.loads(f.read())

    wid2aid = vocab['wid2aid']

    item_boa_file_name = "item_boa.json"

    with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
        vocab = json.loads(f.read())

    with open(os.path.join(args.data_dir, item_boa_file_name), 'r',encoding='utf8') as f:
        item_boa_dict = json.loads(f.read())

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/yelp_restaurant")
    parser.add_argument('--vocab_file', type=str, default="vocab.json")
    parser.add_argument('--attr_file', type=str, default="attr.csv")

    args = parser.parse_args()

