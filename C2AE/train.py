import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from loss import BCE_LOSS, EMBED_LOSS, NORM_LOSS, PAIRWISE_LOSS
from metric import get_precision_recall_F1
from model import _ATTR_NETWORK
from infer_new import _INFER
import random

class _TRAINER(object):

    def __init__(self, vocab, args, device):
        super().__init__()

        self.m_device = device

        # self.m_vocab_size = vocab.vocab_size
        self.m_vocab_size = vocab

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_train_precision = 0
        self.m_mean_train_recall = 0

        self.m_mean_val_loss = 0
        self.m_mean_eval_precision = 0
        self.m_mean_eval_recall = 0
        self.m_mean_eval_F1 = 0
        
        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size
        
        # self.m_rec_loss = _REC_BOW_LOSS(self.m_device)
        # self.m_rec_loss = _REC_SOFTMAX_BOW_LOSS(self.m_device)
        # self.m_rec_loss = _REC_LOSS(self.m_pad_idx, self.m_device)
        # self.m_rec_loss = XE_LOSS(self.m_vocab_size, self.m_device)
        # self.m_rec_loss = BCE_LOSS(self.m_vocab_size, self.m_device)
        self.m_rec_loss = PAIRWISE_LOSS(self.m_device)
        self.m_embed_loss = EMBED_LOSS(self.m_device)
        self.m_norm_loss = NORM_LOSS(self.m_device)

        # self.m_rec_loss = _REC_BPR_LOSS(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval
        self.m_overfit_epoch_threshold = 3

    def f_save_model(self, checkpoint):
        # checkpoint = {'model':network.state_dict(),
        #     'epoch': epoch,
        #     'en_optimizer': en_optimizer,
        #     'de_optimizer': de_optimizer
        # }
        torch.save(checkpoint, self.m_model_file)

    def f_init_word_embed(self, pretrain_word_embed, network):
        network.m_attr_embedding.weight.data.copy_(pretrain_word_embed)
        network.m_attr_embedding.weight.requires_grad = False

    def f_train(self, pretrain_word_embed, train_data, eval_data, network, optimizer, logger_obj):
        last_train_loss = 0
        last_eval_loss = 0

        overfit_indicator = 0

        best_eval_precision = 0
        best_eval_F1 = 0
        # self.f_init_word_embed(pretrain_word_embed, network)
        try: 
            for epoch in range(self.m_epochs):
                
                print("++"*10, epoch, "++"*10)

            
                s_time = datetime.datetime.now()
                # train_data.sampler.set_epoch(epoch)
                self.f_train_epoch(train_data, network, optimizer, logger_obj, epoch)
                e_time = datetime.datetime.now()

                print("epoch duration", e_time-s_time)

                if last_train_loss == 0:
                    last_train_loss = self.m_mean_train_loss

                elif last_train_loss < self.m_mean_train_loss:
                    print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                    # break
                else:
                    print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                    last_train_loss = self.m_mean_train_loss

                print("--"*10, epoch, "--"*10)
                
                s_time = datetime.datetime.now()
                self.f_eval_epoch(eval_data, network, optimizer, logger_obj, epoch)
                e_time = datetime.datetime.now()

                print("validation epoch duration", e_time-s_time)

                # self.m_mean_eval_loss = 1000.0

                if last_eval_loss == 0:
                    last_eval_loss = self.m_mean_eval_loss

                elif last_eval_loss < self.m_mean_eval_loss:
                    print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    
                    overfit_indicator += 1
                else:
                    print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    last_eval_loss = self.m_mean_eval_loss

                if best_eval_F1 < self.m_mean_eval_F1:
                    checkpoint = {'model':network.state_dict()}
                    print("... save model ...")
                    self.f_save_model(checkpoint)
                    best_eval_F1 = self.m_mean_eval_F1

                   
        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early") 
            if best_eval_F1 < self.m_mean_eval_F1:
                print("... saving model ...")
                checkpoint = {'model':network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_F1 = self.m_mean_eval_F1
            
    def f_train_epoch(self, train_data, network, optimizer, logger_obj, epoch_index):
        loss_list = []
        precision_list = []
        recall_list = []

        iteration = 0

        # logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)
        # logger_obj.f_add_output2IO("--"*20)

        tmp_loss_list = []
        tmp_precision_list = []
        tmp_recall_list = []

        network.train()
        alpha = 0.5
        gamma = 0.001
        topk = 3

        batch_index = 0
        for user_batch, item_batch, y_batch in train_data:		
            # print("==="*10)
            # print("x", x_batch)
            # print("y", y_batch)

            batch_size = user_batch.size(0)

            user_gpu = user_batch.to(self.m_device)
            item_gpu = item_batch.to(self.m_device)
            y_gpu = y_batch.to(self.m_device)

            enc_x, enc_y, logits = network(user_gpu, item_gpu, y_gpu, batch_index, epoch_index)

            rec_loss = self.m_rec_loss(logits, y_gpu)

            rec_loss = alpha*rec_loss

            # print("rec_loss", rec_loss)

            embed_loss = self.m_embed_loss(enc_x, enc_y)

            # print("embed_loss", embed_loss)

            norm_loss = self.m_norm_loss(network)

            norm_loss = gamma*norm_loss

            # print("norm_loss", norm_loss)

            loss = rec_loss+embed_loss+norm_loss

            # print("loss", loss)
            # if precision != 0 and recall != 0:
            # precision, recall, F1= get_precision_recall_F1(logits.cpu(), y_batch, k=topk)

            # exit()

            # if batch_index > 2:
            #     exit()

            batch_index += 1

            precision = 1.0
            recall = 1.0

            loss_list.append(loss.item()/batch_size)
            precision_list.append(precision)
            recall_list.append(recall)

            tmp_loss_list.append(loss.item()/batch_size)
            tmp_precision_list.append(precision)
            tmp_recall_list.append(recall)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.m_train_iteration += 1
            
            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%d, NLL_loss:%.4f, precision:%.4f, recall:%.4f"%(iteration, np.mean(tmp_loss_list), np.mean(tmp_precision_list), np.mean(tmp_recall_list)))

                tmp_loss_list = []
                tmp_precision_list = []
                tmp_recall_list = []
            
        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f, precision:%.4f, recall:%.4f"%(self.m_train_iteration, np.mean(loss_list), np.mean(precision_list), np.mean(recall_list)))
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)
        logger_obj.f_add_scalar2tensorboard("train/precision", np.mean(precision_list), self.m_train_iteration)
        logger_obj.f_add_scalar2tensorboard("train/recall", np.mean(recall_list), self.m_train_iteration)

        self.m_mean_train_loss = np.mean(loss_list)
        self.m_mean_train_precision = np.mean(precision_list)
        self.m_mean_train_recall = np.mean(recall_list)

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj, epoch_index):
        loss_list = []
        precision_list = []
        recall_list = []
        F1_list = []

        iteration = 0
        # self.m_eval_iteration = 0
        self.m_eval_iteration = self.m_train_iteration

        # logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)
        # logger_obj.f_add_output2IO("--"*20)

        network.eval()
        topk = 3
        alpha = 0.5
        gamma = 0.001
        with torch.no_grad():
            for user_batch, item_batch, y_batch, y_mask_batch in eval_data:		
            
                user_gpu = user_batch.to(self.m_device) 

                item_gpu = item_batch.to(self.m_device)    

                y_gpu = y_batch.to(self.m_device)        

                y_mask_gpu = y_mask_batch.to(self.m_device)

                logits = network.f_eval_forward(user_gpu, item_gpu, epoch_index)

                # rec_loss = self.m_rec_loss(logits, y_gpu)

                # embed_loss = self.m_embed_loss(enc_x, enc_y)

                # norm_loss = self.m_norm_loss(network)

                # loss = rec_loss+alpha*embed_loss+gamma*norm_loss
                
                precision, recall, F1= get_precision_recall_F1(logits.cpu(), y_batch, y_mask_batch, k=topk)
                
                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)

            logger_obj.f_add_output2IO("%d, precision:%.4f, recall:%.4f, F1:%.4f"%(self.m_eval_iteration, np.mean(precision_list), np.mean(recall_list), np.mean(F1_list)))

            logger_obj.f_add_scalar2tensorboard("eval/precision", np.mean(precision_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)

        self.m_mean_eval_loss = 0.0	
        self.m_mean_eval_precision = np.mean(precision_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        self.m_mean_eval_F1 =np.mean(F1_list)

        network.train()

