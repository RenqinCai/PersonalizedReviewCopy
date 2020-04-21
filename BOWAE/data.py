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

    def f_load_data_amazon(self, args):
        self.m_data_name = args.data_name
        self.m_vocab_file = self.m_data_name+".vocab.json"

        with open(os.path.join(args.data_dir, args.data_file), 'rb') as file:
            data = pickle.load(file)
        
        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r') as file:
            vocab = json.load(file)

        review_corpus = data['review']
        item_corpus = data['item']
        user_num = data['user']

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])
        vocab_obj.f_set_user_num(user_num)
        
        print("vocab size", vocab_obj.m_vocab_size)

        train_data = Amazon(args, vocab_obj, review_corpus['train'], item_corpus)
        valid_data = Amazon(args, vocab_obj, review_corpus['valid'], item_corpus)

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

class Amazon(Dataset):
    def __init__(self, args, vocab_obj, review_corpus, item_corpus):
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
    
        self.m_sample_num = len(review_corpus)
        print("sample num", self.m_sample_num)

        self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        print("batch num", self.m_batch_num)

        ###get length
        
        length_list = []
        self.m_length_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_input_batch_list = [[] for i in range(self.m_batch_num)]
        # self.m_user_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_target_batch_list = [[] for i in range(self.m_batch_num)]
        # self.m_RRe_batch_list = [[] for i in range(self.m_batch_num)]
        # self.m_ARe_batch_list = [[] for i in range(self.m_batch_num)]

        for sample_index in range(self.m_sample_num):
            review_obj = review_corpus[sample_index]
            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len]
            len_review = len(input_review) + 1

            length_list.append(len_review)

        # sorted_length_list = sorted(length_list, reverse=True)
        sorted_index_len_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)

        for i, sample_index in enumerate(sorted_index_len_list):
            batch_index = int(i/self.m_batch_size)
            if batch_index >= self.m_batch_num:
                break
            
            review_obj = review_corpus[sample_index]
            word_ids_review = review_obj.m_review_words

            input_review = word_ids_review[:self.m_max_seq_len] 
            input_review = [self.m_sos_id] + input_review
            target_review = word_ids_review[:self.m_max_seq_len]+[self.m_eos_id]

            self.m_length_batch_list[batch_index].append(length_list[sample_index])

            self.m_input_batch_list[batch_index].append(input_review)
            self.m_target_batch_list[batch_index].append(target_review)

    def __iter__(self):
        print("shuffling")
        
        temp = list(zip(self.m_length_batch_list, self.m_input_batch_list, self.m_target_batch_list))
        random.shuffle(temp)
        
        self.m_length_batch_list, self.m_input_batch_list, self.m_target_batch_list = zip(*temp)

        for batch_index in range(self.m_batch_num):
            # s_time = datetime.datetime.now()

            length_batch = self.m_length_batch_list[batch_index]
            input_batch = self.m_input_batch_list[batch_index]
            target_batch = self.m_target_batch_list[batch_index]

            input_batch_iter = []
            target_batch_iter = []
           
            max_length_batch = max(length_batch)
            # print("max_length_batch", max_length_batch)

            for sent_i, _ in enumerate(input_batch):
                # ss_time = datetime.datetime.now()

                length_i = length_batch[sent_i]
                
                input_i_iter = copy.deepcopy(input_batch[sent_i])
                # target_i_iter = copy.deepcopy(target_batch[sent_i])
                target_i_iter = np.zeros(self.m_vocab_size)
                for word in target_batch[sent_i]:
                    target_i_iter[word] += 1.0
                target_i_word_num = np.sum(target_i_iter)
                target_i_iter /= target_i_word_num
                # print(RRe_index, RRe_val)
                # print(RRe_i_iter[RRe_index])

                input_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))

                input_batch_iter.append(input_i_iter)

                target_i_iter_copy = copy.deepcopy(target_i_iter)

                target_batch_iter.append(target_i_iter_copy)
                # print("target_batch_iter", target_batch_iter)
                # e_time = datetime.datetime.now()
                # print("data batch duration", e_time-ss_time)
                # print(np.sum(target_i_iter), np.sum(target_i_iter_copy))
            # exit()
    
            # print(sum(RRe_batch_iter[0]))
            # ts_time = datetime.datetime.now()
            length_batch_tensor = torch.from_numpy(np.array(length_batch)).long()
            input_batch_iter_tensor = torch.from_numpy(np.array(input_batch_iter)).long()

            target_batch_iter_tensor = torch.from_numpy(np.array(target_batch_iter)).float()

            # print(torch.sum(target_batch_iter_tensor, dim=-1))
            # exit()
            # e_time = datetime.datetime.now()
            # print("tensor data duration", e_time-ts_time)
            # print("data duration", e_time-s_time)
            # print(RRe_batch_iter_tensor.size(), "RRe_batch_iter_tensor", RRe_batch_iter_tensor.sum(dim=1))

            yield input_batch_iter_tensor, target_batch_iter_tensor, length_batch_tensor


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
