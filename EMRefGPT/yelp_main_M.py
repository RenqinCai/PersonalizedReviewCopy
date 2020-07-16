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
from model import _GEN_NETWORK, _ENC_NETWORK
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
from GPT import AdamW, WarmupLinearSchedule, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    seed = 1111
    set_seed(seed)
    #### get data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    args.decoder_model_type = args.decoder_model_type.lower()
    global_step = args.global_step_eval
    
    print("checkpoint dir", args.checkpoint_dir)

    output_decoder_dir = os.path.join(args.checkpoint_dir, "checkpoint-decoder-{}".format(global_step))
    output_full_dir = os.path.join(args.checkpoint_dir, "checkpoint-full-{}".format(global_step))

    print("output_decoder_dir: ", output_decoder_dir)
    print("output_full_dir: ", output_full_dir)

    checkpoints = [[output_decoder_dir]]

    MODEL_CLASSES = {'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)}
    
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]

    model_decoder = decoder_model_class.from_pretrained(output_decoder_dir, latent_size=args.latent_size)

    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
    print("decoder_tokenizer_name ", args.decoder_tokenizer_name)
    print("decoder_model_name_or_path ", args.decoder_model_name_or_path)

    model_decoder.to(device)

    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence
    
    print("max_len_single_sentence: ", tokenizer_decoder.max_len_single_sentence)
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)
    print("block size: ", args.block_size)

    checkpoint = torch.load(os.path.join(output_full_dir, "training.bin"))

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)

    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))

    assert tokenizer_decoder.pad_token == '<PAD>'

    data_obj = _DATA()
    train_data, valid_data, vocab_obj = data_obj.f_load_data_yelp_GPT(tokenizer_decoder, args)
    # train_data, valid_data = data()
    
    if args.train:
        now_time = datetime.datetime.now()
        time_name = str(now_time.month)+"_"+str(now_time.day)+"_"+str(now_time.hour)+"_"+str(now_time.minute)
        model_file = os.path.join(args.model_path, args.data_name+"_"+args.model_name)

        if not os.path.isdir(model_file):
            print("create a directory ", model_file)
            os.mkdir(model_file)

        args.model_file = model_file+"/model_best_"+time_name+".pt"
        print("model_file", model_file)

    network = _GEN_NETWORK(vocab_obj, args)

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
        
        E_network = _ENC_NETWORK(vocab_obj, args)
        E_network = E_network.to(device)
        # E_network = torch.nn.parallel.DistributedDataParallel(E_network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # torch.distributed.barrier()
        # map_location = {'cuda:%d'%0:'cuda:%d'%local_rank}

        model_path = args.model_path
        # E_model_file = args.E_model_file
        # E_model_abs_file = os.path.join(model_path, E_model_file)
        # print("E_model_abs_file", E_model_abs_file)
        
        # check_point = torch.load(E_model_abs_file)

        # check_point = torch.load(E_model_abs_file, map_location=map_location)
        # E_network.load_state_dict(check_point['model'])

        # if args.user_pretrained_model:
        #     pre_model = check_point['model_state_dict']
        #     model_dict = network.state_dict()

        #     pre_dict = {k:v for k, v in pre_model.items() if k in model_dict}
        #     model_dict.update(pre_dict)
        #     network.load_state_dict(model_dict)

        network.init_tokenizer_decoder(tokenizer_decoder, model_decoder)
        network = network.to(device)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in network.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay':args.weight_decay }, 
            {'params': [p for n, p in network.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
        ]
        t_total = len(train_data) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        local_rank = 0
        trainer = _TRAINER(vocab_obj, args, device)
        trainer.f_train_M(train_data, valid_data, E_network, network, optimizer, scheduler, logger_obj, local_rank, args)

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

    parser.add_argument('--local_rank', type=int, help="local rank, necessary for using the torch.distributed.launch utility")

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
    parser.add_argument('--model_name', type=str, default="EMRefSeq")
    parser.add_argument('--hcdmg1', action="store_true", default=False)

    parser.add_argument('--E_model_file', type=str, default="model_best.pt")
    parser.add_argument('--de_strategy', type=str, default="attn")
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--print_interval', type=int, default=400)
    parser.add_argument('--random_flag', type=int, default=0)
    parser.add_argument('--parallel', action="store_true", default=False)
    
    parser.add_argument("--num_train_epochs", default=1.0, type=float,help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="The directory where checkpoints are saved.")
    parser.add_argument("--use_pretrained_model", action='store_true', help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str, help="The decoder model checkpoint for weights initialization.")
    parser.add_argument('--decoder_model_type', type=str, default="gpt2")
    parser.add_argument("--warmup_steps", default=0, type=int,help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument('--global_step_eval', type=int, default=661, help="Evaluate the results at the given global step")
    parser.add_argument('--check_point_dir', type=str, default="")
    parser.add_argument('--decoder_tokenizer_name', type=str, default="")
    parser.add_argument('--do_lower_case', type=str, default="")
    parser.add_argument("--decoder_config_name", default="", type=str, help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument('--block_size', type=int, default=-1)
    parser.add_argument("--max_steps", default=100, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    args = parser.parse_args()

    # args.rnn_type = args.rnn_type.lower()
    # args.anneal_function = args.anneal_function.lower()

    main(args)
