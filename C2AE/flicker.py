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
import random

class FLICKER(Dataset):
    def __init__(self, args):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size

        train_X = self.m_data_dir+"/mirflickr-train-features.pkl"
        train_Y = self.m_data_dir+"/mirflickr-train-labels.pkl"

        X = np.load(train_X, allow_pickle=True)
        Y = np.load(train_Y, allow_pickle=True)

        X = X[0:10417, :]
        Y = Y[0:10417, :]

        num_labels = Y.shape[1]
        full_true = np.ones(num_labels)
        full_false = np.zeros(num_labels)

        i = 0

        while(i < len(Y)):
            if (Y[i] == full_true).all() or (Y[i] == full_false).all():
                Y = np.delete(Y, i, axis=0)
                X = np.delete(X, i, axis=0)
            else:
                i = i+1
            
        self.m_X = X
        self.m_Y = Y

        print("train sample num X", len(self.m_X))

    def __len__(self):
        return len(self.m_X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        x_i = self.m_X[i]

        y_i = self.m_Y[i]

        sample_i = {"x": x_i, "y": y_i}

        return sample_i

    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        x_iter = []

        y_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            x_i = sample_i["x"]
            x_iter.append(x_i)

            y_i = sample_i["y"]
            y_iter.append(y_i)
    
        x_iter_tensor = torch.from_numpy(np.array(x_iter)).float()
        y_iter_tensor = torch.from_numpy(np.array(y_iter)).float()

        return x_iter_tensor, y_iter_tensor
    
class FLICKER_TEST(Dataset):
    def __init__(self, args):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size

        test_X = self.m_data_dir+"/mirflickr-test-features.pkl"
        test_Y = self.m_data_dir+"/mirflickr-test-labels.pkl"

        X = np.load(test_X, allow_pickle=True)
        Y = np.load(test_Y, allow_pickle=True)

        num_labels = Y.shape[1]
        full_true = np.ones(num_labels)
        full_false = np.zeros(num_labels)

        i = 0

        while(i < len(Y)):
            if (Y[i] == full_true).all() or (Y[i] == full_false).all():
                Y = np.delete(Y, i, axis=0)
                X = np.delete(X, i, axis=0)
            else:
                i = i+1

        self.m_X = X
        self.m_Y = Y

        print("test sample num X", len(self.m_X))

    def __len__(self):
        return len(self.m_X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        x_i = self.m_X[i]

        y_i = self.m_Y[i]

        sample_i = {"x": x_i, "y": y_i}

        return sample_i

    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        x_iter = []

        y_iter = []

        y_len_iter = []

        y_mask_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            y_i = sample_i["y"]

            len_i = sum(y_i)
            len_i = int(len_i)
            # print(len_i, end=", ")

            y_len_iter.append(len_i)
        
        max_y_len = max(y_len_iter)
        
        for i in range(batch_size):
            sample_i = batch[i]

            x_i = sample_i["x"]
            x_iter.append(x_i)

            y_i = sample_i["y"]
            # print("y_i", y_i)
            y_len_i = y_len_iter[i]

            new_y_i = list(np.nonzero(y_i)[0])
            # print("new_y_i", new_y_i)

            y_len_i = int(y_len_i)
            len_diff = int(max_y_len-y_len_i)
            new_y_i = new_y_i+[-1]*len_diff

            y_iter.append(new_y_i)
            y_mask_iter.append([1]*y_len_i+[0]*len_diff)

            # if i > 10:
            #     exit()

        x_iter_tensor = torch.from_numpy(np.array(x_iter)).float()
        y_iter_tensor = torch.from_numpy(np.array(y_iter)).long()
        y_mask_iter_tensor = torch.from_numpy(np.array(y_mask_iter)).long()
        
        return x_iter_tensor, y_iter_tensor, y_mask_iter_tensor
    
    
                
        