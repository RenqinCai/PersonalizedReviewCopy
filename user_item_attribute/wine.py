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

class _MOVIE(Dataset):
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
        self.m_input_freq_batch_list = []
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
        attr_list = df.attr.tolist()

        for sample_index in range(self.m_sample_num):
        # for sample_index in range(1000):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = attr_list[sample_index]    
            
            # item_boa = item_boa_dict[str(item_id)]
            # input_boa = item_boa
            # input_boa_freq = [0 for i in range(len(item_boa))]

            # item_boa_boafreq = item_boa_dict[str(item_id)]  
            # item_boa = item_boa_boafreq[0]
            # item_freq = item_boa_boafreq[1]

            item_attrdict_i = item_boa_dict[str(item_id)]  
            item_attr_list_i = list(item_attrdict_i.keys())
            item_attrfreq_list_i = list(item_attrdict_i.values())

            """
            scale the item freq into the range [0, 1]
            """

            def max_min_scale(val_list):
                vals = np.array(val_list)
                min_val = min(vals)
                max_val = max(vals)
                if max_val == min_val:
                    scale_vals = np.zeros_like(vals)
                    # print("scale_vals", scale_vals)
                else:
                    scale_vals = (vals-min_val)/(max_val-min_val)

                scale_vals = scale_vals+1.0
                scale_val_list = list(scale_vals)
                # if max_val-min_val == 0:
                #     print("--"*20)
                #     print("error max_val-min_val", max_val, min_val)
                #     print(item_id, val_list)
                return scale_val_list

            item_freq = max_min_scale(item_attrfreq_list_i)

            input_boa = item_attr_list_i
            input_boa_freq = item_freq

            # target_boa = boa
            target_boa = attrlist_i

            input_len = len(input_boa)
            target_len = len(target_boa)

            self.m_input_batch_list.append(input_boa)
            self.m_input_freq_batch_list.append(input_boa_freq)
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
        input_freq_i = self.m_input_freq_batch_list[i]
        input_length_i = self.m_input_length_batch_list[i]

        user_i = self.m_user_batch_list[i]
        item_i = self.m_item_batch_list[i]

        target_i = self.m_target_batch_list[i]
        target_length_i = self.m_target_length_batch_list[i]
        
        return input_i, input_freq_i, input_length_i, user_i, item_i, target_i, target_length_i, self.m_pad_id, self.m_vocab_size
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        input_iter = []
        input_freq_iter = []
        input_length_iter = []
        user_iter = []
        item_iter = []
        target_iter = []
        target_length_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            input_length_i = sample_i[2]
            input_length_iter.append(input_length_i)

            target_length_i = sample_i[6]
            target_length_iter.append(target_length_i)

        max_input_length_iter = max(input_length_iter)
        max_target_length_iter = max(target_length_iter)

        user_iter = []
        item_iter = []

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)

        for i in range(batch_size):
            sample_i = batch[i]

            input_i = copy.deepcopy(sample_i[0])
            input_i = [int(i) for i in input_i]

            input_freq_i = copy.deepcopy(sample_i[1])
            input_length_i = sample_i[2]

            # if input_i is None:
            #     print("error input is none", sample_i[0])
            # print(input_i)
            # print(len(input_i))

            pad_id = sample_i[7]
            vocab_size = sample_i[8]

            input_i.extend([pad_id]*(max_input_length_iter-input_length_i))
            input_iter.append(input_i)

            input_freq_i.extend([freq_pad_id]*(max_input_length_iter-input_length_i))
            input_freq_iter.append(input_freq_i)

            user_i = sample_i[3]
            user_iter.append(user_i)

            item_i = sample_i[4]
            item_iter.append(item_i)

            # target_i = copy.deepcopy(sample_i[4])
            # target_length_i = sample_i[5]

            # target_i.extend([pad_id]*(max_target_length_iter-target_length_i))
            # target_iter.append(target_i)
            target_index_i = copy.deepcopy(sample_i[5])
            target_i = np.zeros(vocab_size)
            target_i[np.array(target_index_i, int)] = 1

            # print("input_i", input_i)
            # print("target_i", target_i)

            target_i = target_i[input_i]
            target_iter.append(target_i)
        # exit()
        # print("input_iter", input_iter)
        input_iter_tensor = torch.from_numpy(np.array(input_iter)).long()
        input_freq_iter_tensor = torch.from_numpy(np.array(input_freq_iter)).float()
        input_length_iter_tensor = torch.from_numpy(np.array(input_length_iter)).long()
        
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()

        return input_iter_tensor, input_freq_iter_tensor, input_length_iter_tensor, user_iter_tensor, item_iter_tensor, target_iter_tensor

def remove_target_zero_row(args):
    data_dir = args.data_dir
    train_data_file = data_dir + "/train.pickle"
    valid_data_file = data_dir + "/valid.pickle"
    # test_data_file = data_dir+"/test.pickle"

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    # test_df = pd.read_pickle(test_data_file)

    print("columns", train_df.columns)

    print('train num', len(train_df))
    print('valid num', len(valid_df))
    # print('test num', len(test_df))

    item_boa_file = args.item_boa_file

    with open(os.path.join(data_dir, item_boa_file), 'r',encoding='utf8') as f:
        item_boa_dict = json.loads(f.read())

    def remove_target_zero_row_df(df, train_valid_test_flag):
        sample_num = len(df)

        print("=="*10, train_valid_test_flag, "=="*10)
        print("input sample num", sample_num)

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        attrlist_list = df.attr.tolist()
        # rating_list = df.rating.tolist()
        # rating_list = df.rating.tolist()

        data_list = []

        for sample_i in range(sample_num):
            userid_i = userid_list[sample_i]
            itemid_i = itemid_list[sample_i]

            attrlist_i = attrlist_list[sample_i]

            # rating_i = rating_list[sample_i]

            item_boa_boafreq_list = item_boa_dict[str(itemid_i)]
    #         print(item_boa_boafreq_list)
            item_boa = list(item_boa_boafreq_list.keys())
            item_boa = [int(i) for i in item_boa]
            item_boafreq = list(item_boa_boafreq_list.values())
            
    #         print(item_boa)
    #         print(attrlist_i)
            
            common_boa = list(set(attrlist_i) & set(item_boa))

            if len(common_boa) == 0:
                continue

            # boa_i = set(boa_i)
            sub_data = [userid_i, itemid_i, attrlist_i]
            data_list.append(sub_data)

        new_df = pd.DataFrame(data_list)
        print(new_df.head())
        new_df.columns = ['userid', 'itemid', 'attr']
        print("output sample num", len(new_df))

        return new_df

    new_train_data_file = data_dir + "/new_train.pickle"
    new_valid_data_file = data_dir + "/new_valid.pickle"

    new_train_df = remove_target_zero_row_df(train_df, "train")
    new_train_df.to_pickle(new_train_data_file)

    new_valid_df = remove_target_zero_row_df(valid_df, "valid")
    new_valid_df.to_pickle(new_valid_data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/wine_attr_oov")
    parser.add_argument('--item_boa_file', type=str, default="item_attr.json")

    args = parser.parse_args()

    remove_target_zero_row(args)

