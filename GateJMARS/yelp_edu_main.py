import os
import json
import time
import torch
import argparse
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from train import _TRAINER
import torch.nn as nn
from data import _DATA
# from perturb_data_clothing import _Data
# from movie import _MOVIE, _MOVIE_TEST
from clothing import _CLOTHING, _CLOTHING_TEST
from yelp_edu import _YELP, _YELP_TEST
from model import _NETWORK
import datetime
# from inference import INFER
# from inference_new import INFER
from optimizer import _OPTIM
from logger import _LOGGER
import random
# from eval import _EVAL
# from eval_new import _EVAL
from eval_attn import _EVAL
# from eval_attn_prior import _EVAL
from infer_new import _INFER

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    seed = 1111
    set_seed(seed)
    #### get data

    data_obj = _DATA()
    train_data, valid_data, vocab_obj = data_obj.f_load_data_yelp(args)
    # train_data, valid_data = data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    
    if  args.train:
        now_time = datetime.datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.data_name+"_"+args.model_name)

        if not os.path.isdir(model_file):
            print("create a directory ", model_file)
            os.mkdir(model_file)

        args.model_file = model_file+"/model_best_"+time_name+".pt"
        print("model_file", model_file)

    print("vocab_size", vocab_obj.vocab_size)
    print("user num", vocab_obj.user_size)
    ### get model
    network = _NETWORK(vocab_obj, args, device=device)

    ### add count parameters
    total_param_num = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_param_num += param_num
            print(name, "\t", param_num)
    
    print("total parameters num", total_param_num)

    if  args.train:
        logger_obj = _LOGGER()
        logger_obj.f_add_writer(args)

        # if torch.cuda.device_count() > 1:
        #     print("... let us use", torch.cuda.device_count(), "GPUs!")
        #     network = nn.DataParallel(network)

        # print("=="*20)
        # print("device", network.cuda())

            # en_parameters = list(network.module.m_embedding.parameters()) + list(network.module.m_user_item_encoder.parameters()) + list(network.module.m_output2vocab.parameters())
            # en_optimizer = _OPTIM(en_parameters, args)

            # de_parameters = network.module.m_generator.parameters()
            # de_optimizer = _OPTIM(de_parameters, args)

        en_parameters = list(network.m_embedding.parameters()) + list(network.m_user_item_encoder.parameters()) + list(network.m_output2vocab.parameters())
        en_optimizer = _OPTIM(en_parameters, args)

        de_parameters = list(network.m_generator.parameters())
        # de_parameters = list(network.m_embedding.parameters()) + list(network.m_generator.parameters()) + list(network.m_output2vocab.parameters()) +list(network.m_user_embedding.parameters()) + list(network.m_item_embedding.parameters())
        de_optimizer = _OPTIM(de_parameters, args)

        trainer = _TRAINER(vocab_obj, args, device)
        trainer.f_train(train_data, valid_data, network, en_optimizer, de_optimizer, logger_obj)

        logger_obj.f_close_writer()

    if args.test:
        print("="*10, "test", "="*10)
        infer_obj = _INFER(vocab_obj, args, device)

        infer_obj.f_init_infer(network, args.model_file, reload_model=True)

        infer_obj.f_inference(train_data, valid_data)
    
    if args.eval:
        print("="*10, "eval", "="*10)
        
        eval_obj = _EVAL(vocab_obj, args, device)

        eval_obj.f_init_eval(network, args.model_file, reload_model=True)

        eval_obj.f_eval(train_data, valid_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default="yelp_edu")
    parser.add_argument('--data_file', type=str, default="data.pickle")
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=3)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ls', '--latent_size', type=int, default=100)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)

    parser.add_argument('-af', '--anneal_func', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-opt', '--optimizer_type', type=str, default="Adam")
    parser.add_argument('-mp', '--model_path', type=str, default="../checkpoint/")

    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-rnn', '--rnn_type', type=str, default='GRU')
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--layers_num', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('-m', '--momentum', type=float, default=0.00)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--eps', type=float, default=0.0001)
    parser.add_argument('--vocab_file', type=str, default="vocab.json")
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="GateJMARS")
    parser.add_argument('--hcdmg1', action="store_true", default=False)

    parser.add_argument('--de_strategy', type=str, default="attn")
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--print_interval', type=int, default=400)
    parser.add_argument('--random_flag', type=int, default=0)

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    main(args)
