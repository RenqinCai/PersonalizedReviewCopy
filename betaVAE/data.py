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
from BGoogle import BGoogle
from yelp_edu import _YELP, _YELP_TEST
from cloth import _CLOTH, _CLOTH_TEST

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

class _Data():
    def __init__(self):
        print("data")

    def f_create_data(self, args):
        self.m_min_occ = args.min_occ
        self.m_max_line = 1e5

        self.m_data_dir = args.data_dir
        self.m_data_name = args.data_name
        self.m_raw_data_file = args.data_file
        self.m_raw_data_path = os.path.join(self.m_data_dir, self.m_raw_data_file)

        self.m_vocab_file = self.m_data_name+".vocab.json"
        ### to save new generated data
        self.m_data_file = "tokenized_"+self.m_data_name+".pickle"

        data = pd.read_pickle(self.m_raw_data_path)
        train_df = data["train"]
        valid_df = data["valid"]

        tokenizer = TweetTokenizer(preserve_case=False)
        
        train_reviews = train_df.review
        train_item_ids = train_df.itemid
        train_user_ids = train_df.userid

        valid_reviews = valid_df.review
        valid_item_ids = valid_df.itemid
        valid_user_ids = valid_df.userid

        vocab_obj = _Vocab()

        self._create_vocab(vocab_obj, train_reviews)

        review_corpus = defaultdict(dict)
        item_corpus = defaultdict(dict)
        user_corpus = defaultdict(dict)
        user2uid = defaultdict()

        stop_word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in stopwords.words()]
        punc_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in string.punctuation]

        print("loading train reviews")

        ss_time = datetime.datetime.now()

        non_informative_words = stop_word_ids + punc_ids
        print("non informative words num", len(non_informative_words))

        for index, review in enumerate(train_reviews):
            if index > self.m_max_line:
                break

            item_id = train_item_ids.iloc[index]
            user_id = train_user_ids.iloc[index]

            words = tokenizer.tokenize(review)

            word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in words]
            review_id = len(review_corpus['train'])
            review_obj = _Review()
            review_obj.f_set_review(review_id, word_ids, non_informative_words)

            review_corpus["train"][review_id] = review_obj

            if user_id not in user_corpus:
                user_obj = _User()
                user_obj.f_set_user_id(user_id)
                user_corpus[user_id] = user_obj

                user2uid[user_id] = len(user2uid)

            uid = user2uid[user_id]
            user_obj = user_corpus[user_id]
            user_obj.f_add_review_id(review_id)

            if item_id not in item_corpus:
                item_obj = _Item()
                item_corpus[item_id] = item_obj
                item_obj.f_set_item_id(item_id)

            review_obj.f_set_user_item(uid, item_id)

            item_obj = item_corpus[item_id]
            item_obj.f_add_review_id(review_obj, review_id)

        e_time = datetime.datetime.now()
        print("load training duration", e_time-ss_time)

        s_time = datetime.datetime.now()

        user_num = len(user_corpus)
        vocab_obj.f_set_user(user2uid)

        save_item_corpus = {}
        
        print("item num", len(item_corpus))

        print("loading valid reviews")
        for index, review in enumerate(valid_reviews):

            if index > self.m_max_line:
                break

            item_id = valid_item_ids.iloc[index]
            user_id = valid_user_ids.iloc[index]
            
            words = tokenizer.tokenize(review)

            word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in words]

            review_id = len(review_corpus["valid"])

            review_obj = _Review()
            review_obj.f_set_review(review_id, word_ids, non_informative_words)

            review_corpus["valid"][review_id] = review_obj
            
            uid = user2uid[user_id]
            review_obj.f_set_user_item(uid, item_id)

            item_obj = item_corpus[item_id]
            # print(len(item_corpus))
            item_obj.f_get_RRe(review_obj)

        save_data = {"item": save_item_corpus, "review": review_corpus, "user":user_num}

        print("save data to ", self.m_data_file)
        data_pickle_file = os.path.join(self.m_data_dir, self.m_data_file) 
        f = open(data_pickle_file, "wb")
        pickle.dump(save_data, f)
        f.close()

        vocab = dict(w2i=vocab_obj.m_w2i, i2w=vocab_obj.m_i2w, user2uid=vocab_obj.m_user2uid)
        with io.open(os.path.join(self.m_data_dir, self.m_vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

    def f_create_vocab(self, vocab_obj, train_reviews):
        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        # train_reviews = train_df.review
       
        max_line = self.m_max_line
        line_i = 0
        for review in train_reviews:
            words = tokenizer.tokenize(review)
            w2c.update(words)

            if line_i > max_line:
                break

            line_i += 1

        print("max line", max_line)

        for w, c in w2c.items():
            if c > self.m_min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        print("len(i2w)", len(i2w))
        vocab_obj.f_set_vocab(w2i, i2w)

    # def f_load_data_yelp(self, args):

    #     train_data_dir = os.path.join(args.data_dir, 'train.txt')
    #     train_sents = load_sent(train_data_dir)

    #     valid_data_dir = os.path.join(args.data_dir, 'valid.txt')
    #     valid_sents = load_sent(valid_data_dir)

    #     vocab_obj = _Vocab()
        
    #     vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    #     with open(vocab_file) as f:
    #         for line in f:
    #             w = line.split()[0]
    #             wid = len(vocab_obj.m_w2i)

    #             if w == '<go>':
    #                 w = '<sos>'
                
    #             vocab_obj.m_w2i[w] = wid
    #             vocab_obj.m_i2w[wid] = w

    #     voc_size = len(vocab_obj.m_w2i)
    #     vocab_obj.m_vocab_size = voc_size

    #     train_data = Yelp(args, vocab_obj, train_sents)
    #     valid_data = Yelp(args, vocab_obj, valid_sents)

    #     return train_data, valid_data, vocab_obj

    def f_load_data_yelp(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file

        train_data_file = args.data_dir+"/train.pickle"
        valid_data_file = args.data_dir+"/valid.pickle"
        test_data_file = args.data_dir+"/test.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])
        vocab_obj.f_set_user_num(user_num)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _YELP(args, vocab_obj, train_df)
        # valid_data = _YELP_TEST(args, vocab_obj, valid_df)
        valid_data = _YELP(args, vocab_obj, valid_df)


        return train_data, valid_data, vocab_obj

    def f_load_data_google(self, args):

        self.m_data_dir = args.data_dir
        self.m_data_name = args.data_name
        self.m_data_file = args.data_file

        self.m_vocab_file = self.m_data_name+".vocab.json"

        vocab_obj = _Vocab()

        with open(os.path.join(self.m_data_dir, self.m_vocab_file), 'r') as file:
            vocab = json.load(file)
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])
        
        train_data = BGoogle(args, vocab_obj, "train")
        valid_data = BGoogle(args, vocab_obj, "valid")

        return train_data, valid_data, vocab_obj

    def f_load_data_cloth(self, args):
        
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = args.vocab_file

        train_data_file = args.data_dir+"/train.pickle"
        valid_data_file = args.data_dir+"/valid.pickle"
        test_data_file = args.data_dir+"/test.pickle"
        
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        test_df = pd.read_pickle(test_data_file)

        user_num = train_df.userid.nunique()

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r',encoding='utf8') as f:
            vocab = json.loads(f.read())

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])
        vocab_obj.f_set_user_num(user_num)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = _CLOTH(args, vocab_obj, train_df)
        valid_data = _CLOTH(args, vocab_obj, valid_df)

        return train_data, valid_data, vocab_obj

class _Vocab():
    def __init__(self):
        self.m_w2i = {}
        self.m_i2w = {}
        self.m_vocab_size = 0
        self.m_user2uid = {}
        self.m_user_size = 0
    
    def f_set_vocab(self, w2i, i2w):
        self.m_w2i = w2i
        self.m_i2w = i2w
        self.m_vocab_size = self.vocab_size

    def f_set_user(self, user2uid):
        self.m_user2uid = user2uid

    def f_set_user_num(self, user_num):
        self.m_user_size = user_num
    
    @property
    def vocab_size(self):
        return len(self.m_w2i)

    @property
    def pad_idx(self):
        return self.m_w2i['<pad>']

    @property
    def sos_idx(self):
        return self.m_w2i['<sos>']

    @property
    def eos_idx(self):
        return self.m_w2i['<eos>']

    @property
    def unk_idx(self):
        return self.m_w2i['<unk>']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('-dn', '--data_name', type=str, default='amazon')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rnn', type=str, default='GRU')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_file', type=str, default="raw_data.pickle")

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    data_obj = _Data()
    data_obj._create_data(args)
