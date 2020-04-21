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

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    #### get data
    
    # with open(os.path.join(args.data_dir, args.data_file), 'rb') as file:
    #     data = pickle.load(file)

    data_obj = _Data()

    train_data, valid_data, vocab_obj = data_obj.f_load_data_amazon(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger_obj = Logger()
    logger_obj.f_add_writer(args)

    ### add count parameters
    if not args.test:
        now_time = datetime.datetime.now()
        time_name = str(now_time.day)+"_"+str(now_time.month)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.model_name+"/model_best_"+time_name+".pt")
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

    if not args.test:
        optimizer = Optimizer(network.parameters(), args)

        trainer = TRAINER(vocab_obj, args, device)
        trainer.f_train(train_data, valid_data, network, optimizer, logger_obj)

    if args.test:
        print("="*10, "eval", "="*10)
        # eval_obj = EVAL(vocab_obj, args, device)
        # eval_obj.f_init_eval(network, args.model_file, reload_model=True)
        # eval_obj.f_eval(valid_data)
        
        print("="*10, "inference", "="*10)
        
        infer = INFER(vocab_obj, args, device)

        infer.f_init_infer(network, args.model_file, reload_model=True)

        infer.f_inference(valid_data)

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

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)

    parser.add_argument('-af', '--anneal_func', type=str, default='logistic')
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
    parser.add_argument('-md', '--model_name', type=str, default='AEInsertInput')
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--concat', action="store_true", default=False)

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    main(args)
