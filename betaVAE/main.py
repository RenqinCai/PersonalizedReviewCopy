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
import datetime
from inference import INFER
from optimizer import Optimizer
from logger import Logger
from eval import EVAL
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    #### get data

    set_seed(1111)

    data_obj = _Data()

    train_data, valid_data, vocab_obj = data_obj.f_load_data_amazon(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger_obj = Logger()
    logger_obj.f_add_writer(args)

    ### add count parameters
    if args.train:
        now_time = datetime.datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.model_name+"/model_best_"+time_name+"_"+args.data_name+".pt")
        args.model_file = model_file

    print("vocab_size", len(vocab_obj.m_w2i))

    ### get model
    # user_num = 10
    network = REVIEWDI(vocab_obj, args, device=device)

    total_param_num = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_param_num += param_num
            print(name, "\t", param_num)
        
    print("total parameters num", total_param_num)

    if args.train:
        optimizer = Optimizer(network.parameters(), args)
        trainer = TRAINER(vocab_obj, args, device)
        trainer.f_train(train_data, valid_data, network, optimizer, logger_obj)

    if args.test or args.eval:
        print("="*10, "test", "="*10)  
        infer = INFER(vocab_obj, args, device)
        infer.f_init_infer(network, args.model_file, reload_model=True)
        infer.f_inference(valid_data)
    
    if args.eval:
        print("="*10, "eval", "="*10)
        eval_obj = EVAL(vocab_obj, args, device)
        eval_obj.f_init_eval(network, args.model_file, reload_model=True)
        eval_obj.f_eval(valid_data)

        # infer = INFER(vocab_obj, args, device)

        # infer.f_init_infer(network, args.model_file, reload_model=True)

        # input_text = "verrry cheaply constructed , not as comfortable as i expected . i have been wearing this brand , but the first time i wore it , it was a little more than a few days ago . i have been wearing this brand before , so far , no complaints . i will be ordering more in different colors . update : after washing & drying , i will update after washing . after washing , this is a great buy . the fabric is not as soft as it appears to be made of cotton material . <eos>"
        # infer.f_search_text(input_text, train_data)
    
    logger_obj.f_close_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default="amazon")
    parser.add_argument('--data_file', type=str, default="data.pickle")
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=3)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)

    parser.add_argument('-af', '--anneal_func', type=str, default='beta')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-opt', '--optimizer_type', type=str, default="Adam")
    parser.add_argument('-mp', '--model_path', type=str, default="../checkpoint/")

    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-rnn', '--rnn_type', type=str, default='GRU')
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.00)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--eps', type=float, default=0.0001)
    
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('-md', '--model_name', type=str, default='betaVAE')
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--print_interval', type=int, default=400)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--vocab_file', type=str, default="vocab.json")
    parser.add_argument('--var_num', type=int, default=2)

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    main(args)
