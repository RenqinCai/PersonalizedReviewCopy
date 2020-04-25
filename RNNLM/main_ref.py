import os
import json
import time
import torch
import argparse
import numpy as np
# from ptb import PTB, _Data
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from train import TRAINER

from data import Amazon, _Data
from model import REVIEWDI

from inference import INFER
from optimizer import Optimizer
from logger import Logger
from argparse import Namespace
from data_ref import get_data_from_file 

flags = Namespace(
    train_file = "oliver.txt",
    seq_size = 32,
    batch_size = 16,
    embedding_size = 256, 
    lstm_size = 300,
    gradients_norm = 5, 
    initial_words = ['i', "am"],
    predict_top_k = 5,
    checkpoint_path = 'checkpoint',
    model_name = 'RNNLM_ref'
)

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    #### get data
    
    # with open(os.path.join(args.data_dir, args.data_file), 'rb') as file:
    #     data = pickle.load(file)

    # data_obj = _Data()
    # data_obj.f_create_data(args)
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)

    # exit()
    # train_data, valid_data, vocab_obj, user_num= data_obj._load_data_amazon(args)

    # train_data, valid_data, vocab_obj, user_num = data_obj._load_data_amazon(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger_obj = Logger()
    logger_obj.f_add_writer(args)

    ### add count parameters

    print("vocab_size", len(int_to_vocab))

    ### get model
    user_num = 10
    network = REVIEWDI(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size, device=device)

    optimizer = Optimizer(network.parameters(), args)

    trainer = TRAINER(args, flags.seq_size, device)
    
    trainer.f_train(in_text, out_text, network, optimizer, logger_obj)

    # print("*"*10, "inference")
    
    # infer = INFER(vocab_obj, args, device)

    # infer.f_init_infer(network, args.model_file, reload_model=True)

    # infer.f_inference(valid_data)

    logger_obj.f_close_writer()
    # ### get the batch


    ### get the loss


    ### get the backpropogation

    ### save the model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default="amazon")
    parser.add_argument('--data_file', type=str, default="data.pickle")
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=3)

    parser.add_argument('-eb', '--embedding_size', type=int, default=64)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ls', '--latent_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)

    parser.add_argument('-af', '--anneal_func', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-opt', '--optimizer_type', type=str, default="Adam")
    parser.add_argument('-mp', '--model_path', type=str, default="../checkpoint/")

    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-rnn', '--rnn_type', type=str, default='GRU')
    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.00)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--eps', type=float, default=0.0001)
    
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('-md', '--model_name', type=str, default='RNNLM')
    
    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    main(args)
