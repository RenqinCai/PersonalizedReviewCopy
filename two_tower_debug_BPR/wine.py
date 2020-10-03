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

class _WINE(Dataset):
    def __init__(self, args, vocab_obj, df, boa_item_dict, boa_user_dict):
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
        
        self.m_attr_item_list = []
        self.m_attr_tf_item_list = []
        self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        self.m_attr_user_list = []
        self.m_attr_tf_user_list = []
        self.m_attr_length_user_list = []
        self.m_user_batch_list = []
        
        self.m_pos_target_list = []
        self.m_pos_len_list = []

        self.m_neg_target_list = []
        self.m_neg_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        attr_list = df.attr.tolist()

        # print("boa_user_dict", boa_user_dict)

        max_neg_len_threshold = 10000

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])

            attrdict_item_i = boa_item_dict[str(item_id)]              
            attrlist_item_i = list(attrdict_item_i.keys())
            attrlist_item_i = [int(i) for i in attrlist_item_i]
            attrfreq_list_item_i = list(attrdict_item_i.values())

            attrdict_user_i = boa_user_dict[str(user_id)]
            attrlist_user_i = list(attrdict_user_i.keys())
            attrlist_user_i = [int(i) for i in attrlist_user_i]
            attrfreq_list_user_i = list(attrdict_user_i.values())

            attrlist_user_item_i = list(set(attrlist_item_i).union(set(attrlist_user_i)))

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

            attrfreq_list_item_i = max_min_scale(attrfreq_list_item_i)

            self.m_attr_item_list.append(attrlist_item_i)
            self.m_attr_tf_item_list.append(attrfreq_list_item_i)
            self.m_attr_length_item_list.append(len(attrlist_item_i))
            self.m_item_batch_list.append(item_id)
            
            attrfreq_list_user_i = max_min_scale(attrfreq_list_user_i)

            self.m_attr_user_list.append(attrlist_user_i)
            self.m_attr_tf_user_list.append(attrfreq_list_user_i)
            self.m_attr_length_user_list.append(len(attrlist_user_i))
            self.m_user_batch_list.append(user_id)

            self.m_pos_target_list.append(attrlist_i)
            self.m_pos_len_list.append(len(attrlist_i))

            # neg_target_list = list(set(attrlist_user_item_i).difference(set(attrlist_i)))
            # print("attrlist_user_item_i", attrlist_user_item_i)
            # print("attrlist_i", attrlist_i)
            # print("neg target", neg_target_list)
            # exit()
            # neg_target_list = list(set(attrlist_user_item_i).difference(set(attrlist_i)))
            # neg_len = len(neg_target_list)

            neg_target_list = list(set(attrlist_item_i).difference(set(attrlist_i)))
            neg_len = len(neg_target_list)

            if neg_len > max_neg_len_threshold:
                neg_len = max_neg_len_threshold
            self.m_neg_target_list.append(neg_target_list[:neg_len])
            self.m_neg_len_list.append(neg_len)

        print("... load train data ...", len(self.m_attr_item_list), len(self.m_attr_tf_item_list), len(self.m_attr_user_list), len(self.m_attr_tf_user_list))
        # exit()

    def __len__(self):
        return len(self.m_attr_item_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        attr_item_i = self.m_attr_item_list[i]
        attr_tf_item_i = self.m_attr_tf_item_list[i]
        attr_length_item_i = self.m_attr_length_item_list[i]
        
        item_i = self.m_item_batch_list[i]

        attr_user_i = self.m_attr_user_list[i]
        attr_tf_user_i = self.m_attr_tf_user_list[i]
        attr_length_user_i = self.m_attr_length_user_list[i]

        user_i = self.m_user_batch_list[i]

        pos_target_i = self.m_pos_target_list[i]
        pos_len_i = self.m_pos_len_list[i]

        neg_target_i = self.m_neg_target_list[i]
        neg_len_i = self.m_neg_len_list[i]

        sample_i = {"attr_item": attr_item_i, "attr_tf_item": attr_tf_item_i, "attr_length_item": attr_length_item_i, "item": item_i, "attr_user": attr_user_i, "attr_tf_user": attr_tf_user_i, "attr_length_user": attr_length_user_i, "user": user_i,  "pos_target": pos_target_i, "pos_len": pos_len_i, "neg_target": neg_target_i, "neg_len": neg_len_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        attr_item_iter = []
        attr_tf_item_iter = []
        attr_length_item_iter = []
        item_iter = []

        attr_user_iter = []
        attr_tf_user_iter = []
        attr_length_user_iter = []
        user_iter = []

        pos_target_iter = []
        pos_len_iter = []

        neg_target_iter = []
        neg_len_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            attr_length_item_i = sample_i["attr_length_item"]
            attr_length_item_iter.append(attr_length_item_i)

            attr_length_user_i = sample_i["attr_length_user"]
            attr_length_user_iter.append(attr_length_user_i)

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

            neg_len_i = sample_i["neg_len"]
            neg_len_iter.append(neg_len_i)

        max_attr_length_item_iter = max(attr_length_item_iter)
        max_attr_length_user_iter = max(attr_length_user_iter)

        max_pos_targetlen_iter = max(pos_len_iter)
        max_neg_targetlen_iter = max(neg_len_iter)
        
        # print("max_pos_targetlen_iter", max_pos_targetlen_iter)
        # print("max_neg_targetlen_iter", max_neg_targetlen_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            attr_item_i = copy.deepcopy(sample_i["attr_item"])
            attr_item_i = [int(i) for i in attr_item_i]

            attr_tf_item_i = copy.deepcopy(sample_i['attr_tf_item'])
            attr_length_item_i = sample_i["attr_length_item"]
            
            attr_item_i.extend([pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            attr_item_iter.append(attr_item_i)

            attr_tf_item_i.extend([freq_pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            attr_tf_item_iter.append(attr_tf_item_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)

            attr_user_i = copy.deepcopy(sample_i["attr_user"])
            attr_user_i = [int(i) for i in attr_user_i]

            attr_tf_user_i = copy.deepcopy(sample_i['attr_tf_user'])
            attr_length_user_i = sample_i["attr_length_user"]

            attr_user_i.extend([pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            attr_user_iter.append(attr_user_i)

            attr_tf_user_i.extend([freq_pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            attr_tf_user_iter.append(attr_tf_user_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])
            pos_len_i = sample_i["pos_len"]
            pos_target_i.extend([pad_id]*(max_pos_targetlen_iter-pos_len_i))
            pos_target_iter.append(pos_target_i)
            
            neg_target_i = copy.deepcopy(sample_i["neg_target"])
            neg_len_i = sample_i["neg_len"]
            neg_target_i.extend([pad_id]*(max_neg_targetlen_iter-neg_len_i))

            neg_target_iter.append(neg_target_i)

        attr_item_iter_tensor = torch.from_numpy(np.array(attr_item_iter)).long()
        attr_tf_item_iter_tensor = torch.from_numpy(np.array(attr_tf_item_iter)).float()
        attr_length_item_iter_tensor = torch.from_numpy(np.array(attr_length_item_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        attr_user_iter_tensor = torch.from_numpy(np.array(attr_user_iter)).long()
        attr_tf_user_iter_tensor = torch.from_numpy(np.array(attr_tf_user_iter)).float()
        attr_length_user_iter_tensor = torch.from_numpy(np.array(attr_length_user_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        pos_target_iter_tensor = torch.from_numpy(np.array(pos_target_iter)).long()
        pos_len_iter_tensor = torch.from_numpy(np.array(pos_len_iter)).long()

        neg_target_iter_tensor = torch.from_numpy(np.array(neg_target_iter)).long()
        neg_len_iter_tensor = torch.from_numpy(np.array(neg_len_iter)).long()

        return attr_item_iter_tensor, attr_tf_item_iter_tensor, attr_length_item_iter_tensor, item_iter_tensor, attr_user_iter_tensor, attr_tf_user_iter_tensor, attr_length_user_iter_tensor, user_iter_tensor, pos_target_iter_tensor, pos_len_iter_tensor, neg_target_iter_tensor, neg_len_iter_tensor

class _WINE_TEST(Dataset):
    def __init__(self, args, vocab_obj, df, boa_item_dict, boa_user_dict):
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
        
        self.m_attr_item_list = []
        self.m_attr_tf_item_list = []
        self.m_attr_length_item_list = []
        self.m_item_batch_list = []

        self.m_attr_user_list = []
        self.m_attr_tf_user_list = []
        self.m_attr_length_user_list = []
        self.m_user_batch_list = []
        
        self.m_pos_target_list = []
        self.m_pos_len_list = []

        self.m_neg_target_list = []
        self.m_neg_len_list = []

        self.m_user2uid = {}
        self.m_item2iid = {}
        
        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        attr_list = df.attr.tolist()

        # print("boa_user_dict", boa_user_dict)
        max_neg_len_threshold = 10000

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])

            attrdict_item_i = boa_item_dict[str(item_id)]              
            attrlist_item_i = list(attrdict_item_i.keys())
            attrlist_item_i = [int(i) for i in attrlist_item_i]
            attrfreq_list_item_i = list(attrdict_item_i.values())

            attrdict_user_i = boa_user_dict[str(user_id)]
            attrlist_user_i = list(attrdict_user_i.keys())
            attrlist_user_i = [int(i) for i in attrlist_user_i]
            attrfreq_list_user_i = list(attrdict_user_i.values())

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

            attrfreq_list_item_i = max_min_scale(attrfreq_list_item_i)

            self.m_attr_item_list.append(attrlist_item_i)
            self.m_attr_tf_item_list.append(attrfreq_list_item_i)
            self.m_attr_length_item_list.append(len(attrlist_item_i))
            self.m_item_batch_list.append(item_id)
            
            attrfreq_list_user_i = max_min_scale(attrfreq_list_user_i)

            self.m_attr_user_list.append(attrlist_user_i)
            self.m_attr_tf_user_list.append(attrfreq_list_user_i)
            self.m_attr_length_user_list.append(len(attrlist_user_i))
            self.m_user_batch_list.append(user_id)

            self.m_pos_target_list.append(attrlist_i)
            self.m_pos_len_list.append(len(attrlist_i))

            neg_target_list = list(set(attrlist_item_i).difference(set(attrlist_i)))
            neg_len = len(neg_target_list)

            if neg_len > max_neg_len_threshold:
                neg_len = max_neg_len_threshold
            self.m_neg_target_list.append(neg_target_list[:neg_len])
            self.m_neg_len_list.append(neg_len)

        print("... load train data ...", len(self.m_attr_item_list), len(self.m_attr_tf_item_list), len(self.m_attr_user_list), len(self.m_attr_tf_user_list))
        # exit()

    def __len__(self):
        return len(self.m_attr_item_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        attr_item_i = self.m_attr_item_list[i]
        attr_tf_item_i = self.m_attr_tf_item_list[i]
        attr_length_item_i = self.m_attr_length_item_list[i]
        
        item_i = self.m_item_batch_list[i]

        attr_user_i = self.m_attr_user_list[i]
        attr_tf_user_i = self.m_attr_tf_user_list[i]
        attr_length_user_i = self.m_attr_length_user_list[i]

        user_i = self.m_user_batch_list[i]

        pos_target_i = self.m_pos_target_list[i]
        pos_len_i = self.m_pos_len_list[i]

        neg_target_i = self.m_neg_target_list[i]
        neg_len_i = self.m_neg_len_list[i]

        sample_i = {"attr_item": attr_item_i, "attr_tf_item": attr_tf_item_i, "attr_length_item": attr_length_item_i, "item": item_i, "attr_user": attr_user_i, "attr_tf_user": attr_tf_user_i, "attr_length_user": attr_length_user_i, "user": user_i,  "pos_target": pos_target_i, "pos_len": pos_len_i, "neg_target": neg_target_i, "neg_len": neg_len_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        attr_item_iter = []
        attr_tf_item_iter = []
        attr_length_item_iter = []
        item_iter = []

        attr_user_iter = []
        attr_tf_user_iter = []
        attr_length_user_iter = []
        user_iter = []

        pos_target_iter = []
        pos_len_iter = []

        neg_target_iter = []
        neg_len_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            attr_length_item_i = sample_i["attr_length_item"]
            attr_length_item_iter.append(attr_length_item_i)

            attr_length_user_i = sample_i["attr_length_user"]
            attr_length_user_iter.append(attr_length_user_i)

            pos_len_i = sample_i["pos_len"]
            pos_len_iter.append(pos_len_i)

            neg_len_i = sample_i["neg_len"]
            neg_len_iter.append(neg_len_i)

        max_attr_length_item_iter = max(attr_length_item_iter)
        max_attr_length_user_iter = max(attr_length_user_iter)

        max_pos_targetlen_iter = max(pos_len_iter)
        max_neg_targetlen_iter = max(neg_len_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            attr_item_i = copy.deepcopy(sample_i["attr_item"])
            attr_item_i = [int(i) for i in attr_item_i]

            attr_tf_item_i = copy.deepcopy(sample_i['attr_tf_item'])
            attr_length_item_i = sample_i["attr_length_item"]
            
            attr_item_i.extend([pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            attr_item_iter.append(attr_item_i)

            attr_tf_item_i.extend([freq_pad_id]*(max_attr_length_item_iter-attr_length_item_i))
            attr_tf_item_iter.append(attr_tf_item_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)

            attr_user_i = copy.deepcopy(sample_i["attr_user"])
            attr_user_i = [int(i) for i in attr_user_i]

            attr_tf_user_i = copy.deepcopy(sample_i['attr_tf_user'])
            attr_length_user_i = sample_i["attr_length_user"]

            attr_user_i.extend([pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            attr_user_iter.append(attr_user_i)

            attr_tf_user_i.extend([freq_pad_id]*(max_attr_length_user_iter-attr_length_user_i))
            attr_tf_user_iter.append(attr_tf_user_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            pos_target_i = copy.deepcopy(sample_i["pos_target"])
            pos_len_i = sample_i["pos_len"]
            pos_target_i.extend([pad_id]*(max_pos_targetlen_iter-pos_len_i))
            pos_target_iter.append(pos_target_i)
            
            neg_target_i = copy.deepcopy(sample_i["neg_target"])
            neg_len_i = sample_i["neg_len"]
            neg_target_i.extend([pad_id]*(max_neg_targetlen_iter-neg_len_i))

            neg_target_iter.append(neg_target_i)

        attr_item_iter_tensor = torch.from_numpy(np.array(attr_item_iter)).long()
        attr_tf_item_iter_tensor = torch.from_numpy(np.array(attr_tf_item_iter)).float()
        attr_length_item_iter_tensor = torch.from_numpy(np.array(attr_length_item_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()

        attr_user_iter_tensor = torch.from_numpy(np.array(attr_user_iter)).long()
        attr_tf_user_iter_tensor = torch.from_numpy(np.array(attr_tf_user_iter)).float()
        attr_length_user_iter_tensor = torch.from_numpy(np.array(attr_length_user_iter)).long()
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()

        pos_target_iter_tensor = torch.from_numpy(np.array(pos_target_iter)).long()
        pos_len_iter_tensor = torch.from_numpy(np.array(pos_len_iter)).long()

        neg_target_iter_tensor = torch.from_numpy(np.array(neg_target_iter)).long()
        neg_len_iter_tensor = torch.from_numpy(np.array(neg_len_iter)).long()

        return attr_item_iter_tensor, attr_tf_item_iter_tensor, attr_length_item_iter_tensor, item_iter_tensor, attr_user_iter_tensor, attr_tf_user_iter_tensor, attr_length_user_iter_tensor, user_iter_tensor, pos_target_iter_tensor, pos_len_iter_tensor, neg_target_iter_tensor, neg_len_iter_tensor

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
    parser.add_argument('--data_dir', type=str, default="../data/wine_large_attr_oov")
    parser.add_argument('--item_boa_file', type=str, default="item_attr.json")

    args = parser.parse_args()

    remove_target_zero_row(args)

