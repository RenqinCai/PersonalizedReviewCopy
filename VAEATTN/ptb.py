import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import random
from utils import OrderedCounter
import copy

class _Data():
    def __init__(self):
        print("data")

    def f_create_data(self, args):

        self.m_min_occ = args.min_occ
        self.m_max_line = 1e6

        self.m_data_dir = args.data_dir
        self.m_data_name = args.data_name
        self.m_max_seq_len = args.max_seq_length

        # self.m_raw_data_file = args.data_file
        
        self.m_vocab_file = self.m_data_name+"_vocab.json"
        
        ### to save new generated data

        self.m_raw_train_data_path = os.path.join(self.m_data_dir, 'ptb.'+'train'+'.txt')

        vocab_obj = _Vocab()
        self.f_create_vocab(vocab_obj)
        
            # self.f_load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)

        ### create train
        with open(self.m_raw_train_data_path, 'r') as file:

            for i, line in enumerate(file):

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.m_max_seq_len]

                target = words[:self.m_max_seq_len-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                # input.extend(['<pad>'] * (self.max_sequence_length-length))
                # target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in input]
                target = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

                if i > self.m_max_line:
                    break

        self.m_train_data_file = 'ptb.'+'train'+'.json'
        with io.open(os.path.join(self.m_data_dir, self.m_train_data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        ### create validate
        data = defaultdict(dict)
        self.m_raw_validate_data_path = os.path.join(self.m_data_dir, 'ptb.'+'valid'+'.txt')
        self.m_validate_data_file = 'ptb.'+'valid'+'.json'
        with open(self.m_raw_validate_data_path, 'r') as file:

            for i, line in enumerate(file):

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.m_max_seq_len]

                target = words[:self.m_max_seq_len-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                # input.extend(['<pad>'] * (self.max_sequence_length-length))
                # target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in input]
                target = [vocab_obj.m_w2i.get(w, vocab_obj.m_w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

                if i > self.m_max_line:
                    break

        with io.open(os.path.join(self.m_data_dir, self.m_validate_data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

               
    def f_create_vocab(self, vocab_obj):

        # assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.m_raw_train_data_path, 'r') as file:

            max_i = 0

            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)

                max_i = i

                if i > self.m_max_line:
                    break
            
            print("max_i", max_i)

            for w, c in w2c.items():
                if c > self.m_min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.m_data_dir, self.m_vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        print("len(i2w)", len(i2w))
        vocab_obj.f_set_vocab(w2i, i2w)
        # self._load_vocab()

    def f_load_data(self, args):
        ### load vocab data
        self.m_data_name = args.data_name
        self.m_vocab_file = self.m_data_name+"_vocab.json"

        with open(os.path.join(args.data_dir, self.m_vocab_file), 'r') as file:
            vocab = json.load(file)

        vocab_obj = _Vocab()
        vocab_obj.f_set_vocab(vocab['w2i'], vocab['i2w'])

        ### load training data
        train_data = PTB(args, vocab_obj, "train")
        
        ### load validate data
        valid_data = PTB(args, vocab_obj, "valid")

        return train_data, valid_data, vocab_obj

class PTB(Dataset):

    def __init__(self, args, vocab_obj, split):

        super().__init__()
        self.m_data_dir = args.data_dir
        self.m_split = split
        self.m_max_seq_len = args.max_seq_length
        self.m_min_occ = args.min_occ
        self.m_batch_size = args.batch_size

        self.m_max_line = 1e10

        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size

        self.m_data_file = 'ptb.'+self.m_split+'.json'
        with open(os.path.join(args.data_dir, self.m_data_file), 'r') as file:
            self.m_data = json.load(file)

        self.m_max_line = 100000

        self.m_sentence_num = len(self.m_data)
        print("sentence num", self.m_sentence_num)
        self.m_batch_num = int(self.m_sentence_num/self.m_batch_size)
        print("batch num", self.m_batch_num)
        
        length_list = [self.m_data[str(i)]["length"] for i in range(self.m_sentence_num)]
        # print(length_list[:20])
        sorted_index_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)
        # print(len(sorted_index_list))

        self.m_input_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_target_batch_list = [[] for i in range(self.m_batch_num)]
        self.m_length_batch_list = [[] for i in range(self.m_batch_num)]

        for i, sent_i in enumerate(sorted_index_list):
            batch_index = int(i/self.m_batch_size)
            if batch_index >= self.m_batch_num:
                break
            self.m_input_batch_list[batch_index].append(self.m_data[str(sent_i)]["input"])

            self.m_target_batch_list[batch_index].append(self.m_data[str(sent_i)]["target"])

            self.m_length_batch_list[batch_index].append(self.m_data[str(sent_i)]["length"])

    def __len__(self):
        return len(self.m_data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.m_data[idx]['input']),
            'target': np.asarray(self.m_data[idx]['target']),
            'length': self.m_data[idx]['length']
        }

    def __iter__(self):
        print("shuffling")

        temp = list(zip(self.m_input_batch_list, self.m_target_batch_list, self.m_length_batch_list))
        random.shuffle(temp)

        self.m_input_batch_list, self.m_target_batch_list, self.m_length_batch_list = zip(*temp)

        for batch_index in range(self.m_batch_num):
            
            input_batch = self.m_input_batch_list[batch_index]
            target_batch = self.m_target_batch_list[batch_index]
            length_batch = self.m_length_batch_list[batch_index]

            input_batch_iter = []
            target_batch_iter = []

            max_length_batch = max(length_batch)
        
            for sent_i, length_i in enumerate(length_batch):
                input_i = copy.deepcopy(input_batch[sent_i])
                target_i = copy.deepcopy(target_batch[sent_i])
            
                input_i.extend([self.m_pad_id] * (max_length_batch-length_i))
                target_i.extend([self.m_pad_id] * (max_length_batch-length_i))

                input_batch_iter.append(input_i)
                target_batch_iter.append(target_i)

            input_batch_iter = np.array(input_batch_iter)
            input_batch_tensor = torch.from_numpy(input_batch_iter)

            target_batch_iter = np.array(target_batch_iter)
            target_batch_tensor = torch.from_numpy(target_batch_iter)

            length_batch = np.array(length_batch)
            length_batch_tensor = torch.from_numpy(length_batch)

            yield input_batch_tensor, target_batch_tensor, length_batch_tensor

class _Vocab():
    def __init__(self):
        self.m_w2i = None
        self.m_i2w = None
        self.m_vocab_size = 0
        self.m_user2uid = None
        # self.m_user_size = 0
    
    def f_set_vocab(self, w2i, i2w):
        self.m_w2i = w2i
        self.m_i2w = i2w
        self.m_vocab_size = self.vocab_size

    def f_set_user(self, user2uid):
        self.m_user2uid = user2uid
    
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

    