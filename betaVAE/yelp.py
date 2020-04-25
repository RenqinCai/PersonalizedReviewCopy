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

class Yelp(Dataset):
    def __init__(self, args, vocab_obj, review_corpus):
        super().__init__()

        batches, _ = self.get_batches(review_corpus, vocab_obj, args.batch_size)

        self.m_batches = batches

    def get_batch(self, x, vocab):
        go_x, x_eos = [], []
        max_len = max([len(s) for s in x])
        length_x = []
        for s in x:
            s_idx = [vocab.m_w2i[w] if w in vocab.m_w2i else vocab.unk_idx for w in s]
            padding = [vocab.pad_idx] * (max_len - len(s))
            go_x.append([vocab.sos_idx] + s_idx + padding)
            x_eos.append(s_idx + [vocab.eos_idx] + padding)
            length_x.append(len(s)+1)
        
        return torch.LongTensor(go_x).contiguous(), \
            torch.LongTensor(x_eos).contiguous(),\
            torch.LongTensor(length_x).contiguous()
           # time * batch

    def get_batches(self, data, vocab, batch_size):
        order = range(len(data))
        z = sorted(zip(order, data), key=lambda i: len(i[1]))
        order, data = zip(*z)

        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
                j += 1
            batches.append(self.get_batch(data[i: j], vocab))
            i = j
        return batches, order

    def __iter__(self):
        print("shuffling")
        
        # temp = list(zip(self.m_length_batch_list, self.m_input_batch_list, self.m_target_batch_list))
        # random.shuffle(temp)

        indices = list(range(len(self.m_batches)))
        random.shuffle(indices)

        # print("indices", indices)
        
        # self.m_length_batch_list, self.m_input_batch_list, self.m_target_batch_list = zip(*temp)
        for i, idx in enumerate(indices):
            input_batch_iter_tensor, target_batch_iter_tensor, length_batch_tensor = self.m_batches[idx]

        # for batch_index in range(self.m_batch_num):
        #     # s_time = datetime.datetime.now()

            # length_batch = self.m_length_batch_list[batch_index]
            # input_batch = self.m_input_batch_list[batch_index]
            # target_batch = self.m_target_batch_list[batch_index]

            # input_batch_iter = []
            # target_batch_iter = []
           
            # max_length_batch = max(length_batch)
            # # print("max_length_batch", max_length_batch)

            # for sent_i, _ in enumerate(input_batch):
            #     # ss_time = datetime.datetime.now()

            #     length_i = length_batch[sent_i]
                
            #     input_i_iter = copy.deepcopy(input_batch[sent_i])
            #     target_i_iter = copy.deepcopy(target_batch[sent_i])

            #     # print(RRe_index, RRe_val)
            #     # print(RRe_i_iter[RRe_index])

            #     input_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))
            #     target_i_iter.extend([self.m_pad_id]*(max_length_batch-length_i))

            #     input_batch_iter.append(input_i_iter)

            #     target_batch_iter.append(target_i_iter)

            #     # e_time = datetime.datetime.now()
            #     # print("data batch duration", e_time-ss_time)

            # # print(sum(RRe_batch_iter[0]))
            # # ts_time = datetime.datetime.now()
            # length_batch_tensor = torch.from_numpy(np.array(length_batch)).long()
            # input_batch_iter_tensor = torch.from_numpy(np.array(input_batch_iter)).long()

            # target_batch_iter_tensor = torch.from_numpy(np.array(target_batch_iter)).long()

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
