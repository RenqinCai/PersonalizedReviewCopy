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

class _YELP_RESTAURANT(Dataset):
    def __init__(self, args, vocab_obj, df, item_boa_dict, user_boa_dict):
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
            # item_boa = item_boa_dict[str(item_id)]        
            item_boa = item_boa_dict[str(item_id)][0]
            # user_boa = user_boa_dict[str(user_id)][0]

            input_boa = item_boa
            target_boa = boa

            input_len = len(item_boa)
            target_len = len(boa)

            # if not target_boa:
            #     # print("error", target_boa)
            #     continue
            
            # if not input_boa:
            #     # print("error input boa")
            #     continue

            if len(input_boa) == 0:
                print("empty input data", input_boa)
                continue

            if len(target_boa) == 0:
                print("empty target boa", target_boa)
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
        # exit()

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

    print("original attr num", len(attr_list))

    wid2aid = {}
    for w in w2i:
        if w in attr_list:
            wid = w2i[w]
            aid = len(a2i)
            a2i[w] = aid
            i2a[aid] = w
            
            wid = int(wid)
            wid2aid[wid] = aid
        
    print("after merge attr num", len(a2i))
    # exit()
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

    print("reading data")

    train_data_file = data_dir+'/train.pickle'
    valid_data_file = data_dir+'/valid.pickle'
    test_data_file = data_dir+'/test.pickle'

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)

    print("columns", train_df.columns)

    # exit()

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

        boa_freq_map = dict(Counter(boa))
        # boa_list = list(boa_freq_map.keys())
        # boa_freq_list = list(boa_freq_map.values())

        if len(boa) == 0:
            print("error review boa")
        return boa_freq_map

    train_data_file = data_dir+'/new_train.pickle'
    valid_data_file = data_dir+'/new_valid.pickle'
    test_data_file = data_dir+'/new_test.pickle'

    print("--"*20)
    print("training data")
    train_df['boa'] = train_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    train_sample_num = len(train_df)
    print("train_sample_num", train_sample_num)
    # print(len(train_df['boa']) > 0)
    new_train_df = train_df[train_df['boa'].map(len) > 0]
    new_train_sample_num = len(new_train_df)
    print("new train sample num", new_train_sample_num)
    # new_train_df.to_pickle(train_data_file)
    print("--"*20)

    valid_df['boa'] = valid_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    valid_sample_num = len(valid_df)
    print("valid_sample_num", valid_sample_num)
    new_valid_df = valid_df[valid_df['boa'].map(len) > 0]
    new_valid_sample_num = len(new_valid_df)
    print("new valid sample num", new_valid_sample_num)
    # new_valid_df.to_pickle(valid_data_file)
    print("--"*20)

    test_df['boa'] = test_df.apply(lambda row: get_review_boa(row['token_idxs']), axis=1)
    test_sample_num = len(test_df)
    print("test_sample_num", test_sample_num)
    new_test_df = test_df[test_df['boa'].map(len) > 0]
    new_test_sample_num = len(new_test_df)
    print("new test sample num", new_test_sample_num)
    # new_test_df.to_pickle(test_data_file)
    print("--"*20)

    # print(train_df.iloc[0])
    # exit()
    def get_most_freq_val(val_list):
        
        a = Counter({})
        for i in val_list:
            a = a+Counter(i)

        a = dict(a)

        sorted_a = sorted(a.items(), key=lambda item: item[1], reverse=True)
        a = [k for k, v in sorted_a]
        a_freq = [v for k, v in sorted_a]
        
        if len(a) == 0:
            print("error")

        top_k = 100
        top_a = a[:top_k]
        top_a_freq = a_freq[:top_k]

        if len(top_a) == 0:
            print("error item freq")

        return [top_a, top_a_freq]

    item_boa_dict = train_df.groupby('itemid')['boa'].apply(list)
    item_boa_dict = item_boa_dict.apply(lambda row: get_most_freq_val(row))
    item_boa_dict = dict(item_boa_dict)

    user_boa_dict = train_df.groupby('userid')['boa'].apply(list)
    user_boa_dict = user_boa_dict.apply(lambda row: get_most_freq_val(row))
    user_boa_dict = dict(user_boa_dict)

    def remove_target_zero_row(df, train_valid_test_flag):
        sample_num = len(df)
        print("=="*10, train_valid_test_flag, "=="*10)
        print("input sample num", sample_num)

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        review_list = df.review.tolist()
        boa_list = df.boa.tolist()
        # rating_list = df.rating.tolist()

        data_list = []

        for sample_i in range(sample_num):
            userid_i = userid_list[sample_i]
            itemid_i = itemid_list[sample_i]
            review_i = review_list[sample_i]
            boa_map_i = boa_list[sample_i]
            boa_i = list(boa_map_i.keys())
            # rating_i = rating_list[sample_i]

            item_boa_boafreq_list = item_boa_dict[itemid_i]
            item_boa = item_boa_boafreq_list[0]
            item_boafreq = item_boa_boafreq_list[1]
            common_boa = list(set(boa_i) & set(item_boa))

            if len(common_boa) == 0:
                continue

            # boa_i = set(boa_i)
            sub_data = [userid_i, itemid_i, review_i, boa_i]
            data_list.append(sub_data)

        new_df = pd.DataFrame(data_list)
        new_df.columns = ['userid', 'itemid', 'review', 'boa']
        print("output sample num", len(new_df))

        return new_df

    new_train_df = remove_target_zero_row(new_train_df, "train")
    new_train_df.to_pickle(train_data_file)

    new_valid_df = remove_target_zero_row(new_valid_df, "valid")
    new_valid_df.to_pickle(valid_data_file)

    new_test_df = remove_target_zero_row(new_test_df, "test")
    new_test_df.to_pickle(test_data_file)

    item_boa_file_name = "item_boa.json"

    item_boa_abs_file_name = os.path.join(data_dir, item_boa_file_name)
    with open(item_boa_abs_file_name, 'w') as f:
        f.write(json.dumps(item_boa_dict))

    user_boa_file_name = "user_boa.json"
    user_boa_abs_file_name = os.path.join(data_dir, user_boa_file_name)

    with open(user_boa_abs_file_name, 'w') as f:
        f.write(json.dumps(user_boa_dict))

    # item_review_list = train_df.groupby('itemid')['review'].apply(list)
    # item_review_list_dict = dict(item_review_list)
    # for item_id, item_review_list in item_review_list_dict.items():

def pretrain_word2vec(args):
    vocab_file = args.vocab_file
    data_dir = args.data_dir

    print("reading data")

    train_data_file = data_dir+'/new_train.pickle'
    valid_data_file = data_dir+'/new_valid.pickle'
    test_data_file = data_dir+'/new_test.pickle'

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)

    tweet_tokenizer = TweetTokenizer(preserve_case=False)

    def my_tokenizer(s):
        return tweet_tokenizer.tokenize(s)

    train_df['tokens'] = train_df.apply(lambda row: my_tokenizer(row['review']), axis=1)
    corpus = train_df.tokens.tolist()
    corpus_size = len(corpus)
    print("corpus size", corpus_size)

    print("--"*10, "training model", "--"*10)
    model = gensim.models.Word2Vec(corpus, min_count=5, size=300, workers=3, window=5, sg=1)
    
    print("--"*10, "save model", "--"*10)
    model_file = data_dir+"/skip_word2vec.model"
    model.save(model_file)
    print("--"*10, "!done!", "--"*10)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/yelp_restaurant")
    parser.add_argument('--vocab_file', type=str, default="vocab.json")
    parser.add_argument('--attr_file', type=str, default="attr.csv")

    args = parser.parse_args()

    pretrain_word2vec(args)

    # f_merge_attr(args)
    # f_get_bow_item(args)