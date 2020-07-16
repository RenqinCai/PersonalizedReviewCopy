import numpy as np
import os, sys
import torch
from torch import nn, optim
import subprocess
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
import json
import pdb
import pandas as pd
import torch.nn.init as init

import glob
import logging
import pickle
import random
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]

        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)

class BucketingDataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, tokenizer, args, bucket=100, shuffle=True):

        self.dataset = TokenDataset(tokenizer, args, file_path, block_size=args.block_size)
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['gpt2_token_length'] for example in self.dataset.examples]

    def __iter__(self):
        sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=TokenDataset.collate)
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass

class TokenDataset(Dataset):
    def __init__(self, tokenizers, args, file_path='train', text_split_mode='natural', block_size=512):
        print("file path", file_path)

        assert os.path.isfile(file_path)

        df = pd.read_pickle(file_path)
        # print("==="*20)
        # print(df.head())
        # print("==="*20)
        # directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(args.data_dir, f'cached_lm_gpt_{block_size}_{args.data_name[:-4]}.json')

        self.examples = []
        self.tokenizers = tokenizers

        print("tokenizers num", len(self.tokenizers))

        # Bert tokenizer special tokens
        # self.bert_pad_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].pad_token])[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].pad_token])[0]
        self.gpt2_bos_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].bos_token])[0]
        self.gpt2_eos_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].eos_token])[0]

        global gpt2_pad_token
        gpt2_pad_token = self.gpt2_pad_token

        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'r') as handle:
                self.examples = json.load(handle)

        else:
            dropped, count = self._read_corpus_natural_split(df, max_length=block_size, block_size=block_size, args=args)
            
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            with open(cached_features_file, 'w') as handle:
                json.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        userid = [torch.tensor(f['userid']).long() for f in examples]
        itemid = [torch.tensor(f['itemid']).long() for f in examples]
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[f['gpt2_token_length']] for f in examples] , dtype=torch.long)

        return (userid, itemid, input_ids_gpt, token_lengths)

    def _read_corpus_natural_split(self, data, max_length, block_size, args):
    
        dropped = 0
        count = 0

        userid_list = data.userid.tolist()
        itemid_list = data.itemid.tolist()
        review_list = data.review.tolist()

        example_num = len(userid_list)
        for example_id in range(example_num):
            userid = userid_list[example_id]
            itemid = itemid_list[example_id]

            review = review_list[example_id]
            split_line_text = review
       
            if len(split_line_text.split()) < 1:
                dropped += 1
                continue

            if max_length:
                if len(split_line_text.split()) > max_length:
                    dropped += 1
                    continue

            tokenized_text1 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(split_line_text))
            tokenized_text1 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text1)
            tokenized_text1 = [self.gpt2_bos_token] + tokenized_text1 + [self.gpt2_eos_token]
            tokenized_text1_length = len(tokenized_text1)

            example = {
                'userid': userid,
                'itemid': userid,
                'gpt2_token':tokenized_text1,
                'gpt2_token_length': tokenized_text1_length
            }
            self.examples.append(example)
            count +=1

        return dropped, count