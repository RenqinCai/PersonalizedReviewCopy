from datetime import datetime
import numpy as np
import torch
import random
import torch.nn as nn
import torch.cuda

import pickle
import argparse
from data import _DATA
import json
import os
from optimizer import _OPTIM
from logger import _LOGGER
# import matplotlib.pyplot as plt
import time
from train import _TRAINER
from model import _ATTR_NETWORK
from eval_attn import _EVAL
from infer_attn import _INFER
# from eval_pop import _EVAL
# from eval_user_item_pop import _EVAL

# from pytorch_visualize import *         

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    seed = 1111
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    data_obj = _DATA()
    if "yelp" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_data_yelp_restaurant(args)

    if "movie" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_data_movie(args)

    if "beer" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_data_beer(args)

    if "wine" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_data_wine(args)

    if "lthing" in args.data_name:
        train_data, valid_data, vocab_obj = data_obj.f_load_data_movie(args)

    if args.train:
        now_time = datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.data_name+"_"+args.model_name)

        if not os.path.isdir(model_file):
            print("create a directory", model_file)
            os.mkdir(model_file)

        args.model_file = model_file+"/model_best_"+time_name+".pt"
        print("model_file", model_file)
    
    print("vocab_size", vocab_obj.vocab_size)
    print("user num", vocab_obj.user_num)
    print("item num", vocab_obj.item_num)

    pretrain_word_embed = None
    # pretrain_word_embed = load_pretrain_word2vec(vocab_obj, args)
    # pretrain_word_embed = pretrain_word_embed.to(device)
    
    network = _ATTR_NETWORK(vocab_obj, args, device)

    total_param_num = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_param_num += param_num
            print(name, "\t", param_num)

    print("total parameters num", total_param_num)

    if args.train:
        logger_obj = _LOGGER()
        logger_obj.f_add_writer(args)

        optimizer = _OPTIM(network.parameters(), args)
        trainer = _TRAINER(vocab_obj, args, device)
        trainer.f_train(pretrain_word_embed, train_data, valid_data, network, optimizer, logger_obj)

        logger_obj.f_close_writer()

    if args.eval:
        print("="*10, "eval", "="*10)
        
        eval_obj = _EVAL(vocab_obj, args, device)

        network = network.to(device)

        eval_obj.f_init_eval(network, args.model_file, reload_model=True)

        # eval_obj.f_eval_new_user(train_data, valid_data)
        eval_obj.f_eval(train_data, valid_data)

    if args.test:
        print("="*10, "eval", "="*10)
        
        infer_obj = _INFER(vocab_obj, args, device)

        network = network.to(device)

        infer_obj.f_init_infer(network, args.model_file, reload_model=True)

        infer_obj.f_infer(train_data, valid_data)

    # emb = network.m_decoder.weight.data
    # # print("emb size 0", emb.shape)
    # emb = emb.cpu().numpy().T
    # print("emb size 1", emb.shape)

    # print_top_words(emb, list(zip(*sorted(vocab_obj.w2i.items(), key=lambda x:x[1])))[0])
    # # print_perp(model)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='yelp_edu')
    parser.add_argument('--data_file', type=str, default='data.pickle')
    
    parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--item_boa_file', type=str, default='item_boa.json')
    parser.add_argument('--user_boa_file', type=str, default='user_boa.json')
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="tower_user_attribute")
    parser.add_argument('--model_path', type=str, default="../checkpoint/")

    ### model
    parser.add_argument('--attr_emb_size', type=int, default=300)
    parser.add_argument('--user_emb_size', type=int, default=300)
    parser.add_argument('--item_emb_size', type=int, default=300)
    
    parser.add_argument('--attn_head_num', type=int, default=4)
    parser.add_argument('--attn_layer_num', type=int, default=2)
    parser.add_argument('--output_hidden_size', type=int, default=300)

    ### train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--hcdmg1', action="store_true", default=False)
    
    ### hyper-param
    # parser.add_argument('--init_mult', type=float, default=1.0)
    # parser.add_argument('--variance', type=float, default=0.995)
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument

    ### others
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--parallel', action="store_true", default=False)

    args = parser.parse_args()

    main(args)


