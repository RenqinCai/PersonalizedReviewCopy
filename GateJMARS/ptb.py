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

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)
        self.batch_size = kwargs.get('batch_size', 32)

        self.raw_data_path = os.path.join(data_dir, 'ptb.'+split+'.txt')
        self.data_file = 'ptb.'+split+'.json'
        self.vocab_file = 'ptb.vocab.json'

        self.max_line = 100000

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        
        self.sentence_num = len(self.data)
        print("sentence num", self.sentence_num)
        self.batch_num = int(self.sentence_num/self.batch_size)
        print("batch num", self.batch_num)
        
        length_list = [self.data[str(i)]["length"] for i in range(self.sentence_num)]
        print(length_list[:20])
        sorted_index_list = sorted(range(len(length_list)), key=lambda k: length_list[k], reverse=True)
        print(len(sorted_index_list))

        self.input_batch_list = [[] for i in range(self.batch_num)]
        self.target_batch_list = [[] for i in range(self.batch_num)]
        self.length_batch_list = [[] for i in range(self.batch_num)]

        for i, sent_i in enumerate(sorted_index_list):
            batch_index = int(i/self.batch_size)
            if batch_index >= self.batch_num:
                break
            self.input_batch_list[batch_index].append(self.data[str(sent_i)]["input"])

            self.target_batch_list[batch_index].append(self.data[str(sent_i)]["target"])

            self.length_batch_list[batch_index].append(self.data[str(sent_i)]["length"])

        for batch_index in range(self.batch_num):
            input_batch = self.input_batch_list[batch_index]
            target_batch = self.target_batch_list[batch_index]
            length_batch = self.length_batch_list[batch_index]

            max_length_batch = max(length_batch)
            
            for sent_i, input_i in enumerate(input_batch):
                length_i = length_batch[sent_i]
                target_i = target_batch[sent_i]
                # print("length_i", length_i)
                # print("input_i", input_i)
                input_i.extend([self.w2i['<pad>']] * (max_length_batch-length_i))
                target_i.extend([self.w2i['<pad>']] * (max_length_batch-length_i))

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                # input.extend(['<pad>'] * (self.max_sequence_length-length))
                # target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

                if i > self.max_line:
                    break

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def __iter__(self):
        print("shuffling")

        temp = list(zip(self.input_batch_list, self.target_batch_list, self.length_batch_list))
        random.shuffle(temp)

        self.input_batch_list, self.target_batch_list, self.length_batch_list = zip(*temp)

        for batch_index in range(self.batch_num):
            # input_batch = self.input_batch_list[batch_index]
            # target_batch = self.target_batch_list[batch_index]
            # length_batch = self.length_batch_list[batch_index]

            # max_length_batch = max(length_batch)
            # print("max_length_batch", max_length_batch)

            # for sent_i, input_i in enumerate(input_batch):
            #     length_i = length_batch[sent_i]
            #     target_i = target_batch[sent_i]
                # print("length_i", length_i)
                # print("input_i", input_i)
                # input_i.extend([self.w2i['<pad>']] * (max_length_batch-length_i))
                # target_i.extend([self.w2i['<pad>']] * (max_length_batch-length_i))

                # print("input_i", len(input_i))

            # print("shape", len(input_batch), len(input_batch[0]))
            input_batch = self.input_batch_list[batch_index]
            target_batch = self.target_batch_list[batch_index]
            length_batch = self.length_batch_list[batch_index]

            input_batch_tensor = torch.LongTensor(input_batch)
            target_batch_tensor = torch.LongTensor(target_batch)
            length_batch_tensor = torch.LongTensor(length_batch)

            yield input_batch_tensor, target_batch_tensor, length_batch_tensor

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            max_i = 0

            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)

                max_i = i

                if i > self.max_line:
                    break
            
            print("max_i", max_i)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
