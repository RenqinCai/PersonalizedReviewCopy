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

        self.m_output_file = args.output_file
        # self.m_vocab_file = self.m_data_name+".vocab.json"
        self.m_vocab_file = "vocab.json"
        ### to save new generated data
        self.m_data_file = "tokenized_"+self.m_output_file
        # self.m_data_file = "tokenized_"+self.m_data_name+"_"+self.m_output_file
        # self.m_data_file = "tokenized_"+self.m_data_name+"_pro_v2.pickle"

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

        self.f_create_vocab(vocab_obj, train_reviews)
        # i = 0

        review_corpus = defaultdict(dict)
        item_corpus = defaultdict(dict)
        user_corpus = defaultdict(dict)
        global_user2uid = defaultdict()
        global_item2iid = defaultdict()

        stop_word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in stopwords.words('english')]
        punc_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in string.punctuation]

        print("loading train reviews")

        ss_time = datetime.datetime.now()

        non_informative_words = stop_word_ids + punc_ids
        # non_informative_words = stopwords.words()+string.punctuation
        print("non informative words num", len(non_informative_words))

        ### load user words
        self.m_user_word_file = os.path.join(self.m_data_dir, args.user_word_file)
        self.f_load_user_word_score(vocab_obj, self.m_user_word_file)

        ### load item words
        self.m_item_word_file = os.path.join(self.m_data_dir, args.item_word_file)
        self.f_load_item_word_score(vocab_obj, self.m_item_word_file)

        print("user word", len(self.m_user_word_score_map))
        print("item word", len(self.m_item_word_score_map))

        for index, review in enumerate(train_reviews):
            if index > self.m_max_line:
                break

            item_id = train_item_ids.iloc[index]
            user_id = train_user_ids.iloc[index]

            words = tokenizer.tokenize(review)
            
            word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in words]

            new_review_user, new_review_item, new_review_local = self.f_get_perturb(vocab_obj, item_id, words, word_ids)

            review_id = len(review_corpus['train'])
            review_obj = _Review()

            review_obj.f_set_review(review_id, word_ids)
            # review_obj.f_set_review(review_id, word_ids, new_word_tf_map, informative_word_num)
            review_obj.f_set_pertub_review(new_review_user, new_review_item, new_review_local)

            review_corpus["train"][review_id] = review_obj

            if user_id not in user_corpus:
                user_obj = _User()
                user_obj.f_set_user_id(user_id)
                user_corpus[user_id] = user_obj

                global_user2uid[user_id] = len(global_user2uid)

            uid = global_user2uid[user_id]
            user_obj = user_corpus[user_id]
            user_obj.f_add_review_id(review_id)

            if item_id not in item_corpus:
                item_obj = _Item()
                item_corpus[item_id] = item_obj
                item_obj.f_set_item_id(item_id)

                global_item2iid[item_id] = len(global_item2iid)
            
            iid = global_item2iid[item_id]
            item_obj = item_corpus[item_id]
            item_obj.f_add_review_id(review_obj, review_id)

            review_obj.f_set_user_item(uid, iid)

        e_time = datetime.datetime.now()
        print("load training duration", e_time-ss_time)
        print("load train review num", len(review_corpus["train"]))

        s_time = datetime.datetime.now()

        user_num = len(user_corpus)
        vocab_obj.f_set_user(global_user2uid)
        
        print("item num", len(item_corpus))

        print("loading valid reviews")
        for index, review in enumerate(valid_reviews):

            if index > self.m_max_line:
                break

            item_id = valid_item_ids.iloc[index]
            user_id = valid_user_ids.iloc[index]

            if user_id not in global_user2uid:
                continue
            
            if item_id not in item_corpus:
                continue
            
            words = tokenizer.tokenize(review)

            word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in words]

            new_review_user, new_review_item, new_review_local = self.f_get_perturb(vocab_obj, item_id, words, word_ids)

            review_id = len(review_corpus["valid"])
            review_obj = _Review()

            review_obj.f_set_review(review_id, word_ids)
            # review_obj.f_set_review(review_id, word_ids, new_word_tf_map, informative_word_num)
            review_obj.f_set_pertub_review(new_review_user, new_review_item, new_review_local)

            review_corpus["valid"][review_id] = review_obj
            
            uid = global_user2uid[user_id]
            iid = global_item2iid[item_id]
            review_obj.f_set_user_item(uid, iid)

            item_obj = item_corpus[item_id]
            # print(len(item_corpus))
            # item_obj.f_get_RRe(review_obj)
            # item_obj.f_get_ARe(review_obj)

        print("load validate review num", len(review_corpus["valid"]))

        save_data = {"item": global_item2iid, "review": review_corpus, "user":global_user2uid}

        print("save data to ", self.m_data_file)
        data_pickle_file = os.path.join(self.m_data_dir, self.m_data_file) 
        f = open(data_pickle_file, "wb")
        pickle.dump(save_data, f)
        f.close()

        vocab = dict(w2i=vocab_obj.m_w2i, i2w=vocab_obj.m_i2w, user2uid=vocab_obj.m_user2uid)
        with io.open(os.path.join(self.m_data_dir, self.m_vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

    def f_get_perturb(self, vocab_obj, item_id, tokens, token_ids):
        tokenizer = TweetTokenizer(preserve_case=False)
        seperator = " "
        candidate_user_markers = []
        candidate_item_markers = []
        # candidate_local_markers = []
        # new_tokens = [str(i) for i in tokens]

        n_grams = []
        for i in range(1, 4):
            i_grams = [seperator.join(gram) for gram in ngrams(tokens, i)]
            n_grams.extend(i_grams)

        for gram in n_grams:
            if gram in self.m_user_word_score_map:
                candidate_user_markers.append((gram, self.m_user_word_score_map[gram]))
            
            if gram in self.m_item_word_score_map[item_id]:
                candidate_item_markers.append((gram, self.m_item_word_score_map[item_id][gram]))
        
        candidate_user_markers.sort(key=lambda x:x[1], reverse=True)
        candidate_user_markers = [marker for (marker, score) in candidate_user_markers]
        new_review_user = seperator.join(tokens)

        local_marker_list = []
        replace_user_marker_list = []
        for marker in candidate_user_markers:
            # print("user marker", marker)
            if marker in new_review_user:
                new_review_user = new_review_user.replace(marker, "")
                local_marker_list.append(marker)
                
                replace_user_marker_list.append(marker)
        
        candidate_item_markers.sort(key=lambda x:x[1], reverse=True)
        candidate_item_markers = [marker for (marker, score) in candidate_item_markers]
        new_review_item = seperator.join(tokens)
        for marker in candidate_item_markers:
            # print("item marker", marker)
            if marker in new_review_item:
                new_review_item = new_review_item.replace(marker, "")
                local_marker_list.append(marker)
                    # remove_item_marker_list.append(marker)

        new_review_local = seperator.join(local_marker_list)
        
        new_review_user = tokenizer.tokenize(new_review_user)
        user_word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in new_review_user]

        new_review_item = tokenizer.tokenize(new_review_item)
        item_word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in new_review_item]

        new_review_local = tokenizer.tokenize(new_review_local)
        local_word_ids = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in new_review_local]

        user_word_ids = [int(i) for i in user_word_ids]
        item_word_ids = [int(i) for i in item_word_ids]
        local_word_ids = [int(i) for i in local_word_ids]

        if len(user_word_ids) == 0:
            user_word_ids = [random.choice(token_ids)]

        if len(item_word_ids) == 0:
            item_word_ids = [random.choice(token_ids)]
            
        if len(local_word_ids) == 0:
            local_word_ids = [random.choice(token_ids)]

        return user_word_ids, item_word_ids, local_word_ids
        
    def f_load_user_word_score(self, vocab_obj, word_score_file):
        f = open(word_score_file, "rb")
        word_score_map = pickle.load(f)
        
        print("init user word num", len(word_score_map))

        self.m_user_word_score_map = word_score_map
        
    def f_load_item_word_score(self, vocab_obj, word_score_file):
        f = open(word_score_file, "rb")
        item_word_score_map = pickle.load(f)

        print("init item num", len(item_word_score_map))

        # self.m_item_word_score_map = defaultdict(dict)
        self.m_item_word_score_map = item_word_score_map
        # for item in item_word_score_map:
        #     # print("item init word num", len(item_word_score_map[item])

        #     self.m_item_word_score_map[item] = dict()
        #     word_score_map = item_word_score_map[item]
        #     for word_grams in word_score_map:
        #         word_score = float(word_score_map[word_grams])
        #         break_flag = True
        #         word_idx_grams = []

        #         splitted_word_grams = word_grams.split(" ")
        #         for word in splitted_word_grams:
        #             if word in vocab_obj.m_w2i:
        #                 word_idx = vocab_obj.m_w2i[word]
        #                 word_idx_grams.append(str(word_idx))
        #                 break_flag = False
        #             else:
        #                 break_flag = True
        #                 break

        #         if not break_flag:
        #             word_idx_grams = " ".join(word_idx_grams)
                    
        #             if word_idx_grams in self.m_user_word_score_map:
        #                 continue
                    
        #             self.m_item_word_score_map[item][word_idx_grams] = word_score

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

        print("threshold max line", max_line, "load max line", min(line_i, max_line))

        for w, c in w2c.items():
            if c > self.m_min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        print("len(i2w)", len(i2w))
        vocab_obj.f_set_vocab(w2i, i2w)

    def f_load_data(self, args):
        self.m_data_name = args.data_name
        # self.m_vocab_file = self.m_data_name+"_vocab.json"
        self.m_vocab_file = "vocab.json"
        print("data_dir", args.data_dir)

        with open(os.path.join(args.data_dir, args.data_file), 'rb') as file:
            data = pickle.load(file)
        
        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r') as file:
            vocab = json.load(file)

        review_corpus = data['review']
        
        global_user2iid = data['user']
        global_item2iid = data['item']

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])
        
        # item_num = len(item_corpus)

        vocab_obj.f_set_user(global_user2iid)
        vocab_obj.f_set_item(global_item2iid)
        # print("vocab size", vocab_obj.m_vocab_size)

        train_data = _CLOTHING(args, vocab_obj, review_corpus['train'])
        # valid_data = Amazon(args, vocab_obj, review_corpus['valid'])
        valid_data = _CLOTHING_TEST(args, vocab_obj, review_corpus['valid'])
        # valid_data = _CLOTHING(args, vocab_obj, review_corpus['valid'])

        # train_data = _MOVIE(args, vocab_obj, review_corpus['train'])
        # valid_data = _MOVIE_TEST(args, vocab_obj, review_corpus['valid'])

        return train_data, valid_data, vocab_obj

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
        self.m_user_size = len(self.m_user2uid)

    def f_set_user_num(self, user_num):
        self.m_user_size = user_num

    def f_set_item(self, item2iid):
        self.m_item2iid = item2iid
        self.m_item_size = len(item2iid)

    @property
    def item_size(self):
        return self.m_item_size

    @property
    def user_size(self):
        return self.m_user_size

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
