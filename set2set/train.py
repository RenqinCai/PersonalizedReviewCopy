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
from loss import _BPR_LOSS, _REC_BPR_LOSS
from metric import get_precision_recall, get_precision_recall_train, get_precision_recall_F1, get_precision_recall_F1_set
# from model_debug import _ATTR_NETWORK
# from model import _ATTR_NETWORK
from model_set import _ATTR_NETWORK
# from model_user_set_transformer import _ATTR_NETWORK
# from model import _ATTR_NETWORK
from infer_new import _INFER
import random

class _TRAINER(object):

    def __init__(self, vocab, args, device):
        super().__init__()

        self.m_device = device

        self.m_pad_idx = vocab.pad_idx

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
        # self.m_rec_loss = _REC_BPR_LOSS(self.m_device)
        self.m_rec_loss = _BPR_LOSS(self.m_device)

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

    def f_init_model(self, pretrain_network, network):
        network.m_user_embedding.weight.data.copy_(pretrain_network.m_user_embedding.weight.data)
        network.m_item_embedding.weight.data.copy_(pretrain_network.m_item_embedding.weight.data)

        network.m_output_attr_embedding_user.weight.data.copy_(pretrain_network.m_output_attr_embedding_user.weight.data)
        network.m_output_attr_embedding_item.weight.data.copy_(pretrain_network.m_output_attr_embedding_item.weight.data)

        # network.m_user_embedding.weight.requires_grad = False
        # network.m_item_embedding.weight.requires_grad = False
        # network.m_output_attr_embedding_user.weight.requires_grad = False
        # network.m_output_attr_embedding_item.weight.requires_grad = False


    def f_train(self, pretrain_network, train_data, eval_data, network, optimizer, logger_obj):
        last_train_loss = 0
        last_eval_loss = 0

        overfit_indicator = 0

        best_eval_precision = 0
        best_eval_F1 = 0
        best_eval_loss = 0
        
        if pretrain_network is not None:
            self.f_init_model(pretrain_network, network)

        try: 
            for epoch in range(self.m_epochs):
                
                print("++"*10, epoch, "++"*10)

                s_time = datetime.datetime.now()
                self.f_eval_epoch(eval_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()

                print("validation epoch duration", e_time-s_time)

                if last_eval_loss == 0:
                    last_eval_loss = self.m_mean_eval_loss

                elif last_eval_loss < self.m_mean_eval_loss:
                    print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    
                    overfit_indicator += 1

                    # if overfit_indicator > self.m_overfit_epoch_threshold:
                    # 	break
                else:
                    print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
                    last_eval_loss = self.m_mean_eval_loss

                if best_eval_loss < self.m_mean_eval_loss:
                    checkpoint = {'model':network.state_dict()}
                    print("... save model ...")
                    self.f_save_model(checkpoint)
                    best_eval_loss = self.m_mean_eval_loss

                print("--"*10, epoch, "--"*10)

                s_time = datetime.datetime.now()
                # train_data.sampler.set_epoch(epoch)
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
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

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early") 
            if best_eval_F1 < self.m_mean_eval_F1:
                print("... saving model ...")
                checkpoint = {'model':network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_F1 = self.m_mean_eval_F1
            
    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
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

        for pos_attr_set_batch, pos_attr_len_batch, neg_attr_set_batch, neg_attr_len_batch, negtarget_num_batch, user_batch, item_batch in train_data:
            
            pos_attr_set_gpu = pos_attr_set_batch.to(self.m_device)
            pos_attr_len_gpu = pos_attr_len_batch.to(self.m_device)

            neg_attr_set_gpu = neg_attr_set_batch.to(self.m_device)
            neg_attr_len_gpu = neg_attr_len_batch.to(self.m_device)
            negtarget_num_gpu = negtarget_num_batch.to(self.m_device)

            user_gpu = user_batch.to(self.m_device)

            item_gpu = item_batch.to(self.m_device)

            logits = network(pos_attr_set_gpu, pos_attr_len_gpu, neg_attr_set_gpu, neg_attr_len_gpu, negtarget_num_gpu, user_gpu, item_gpu)

            NLL_loss = self.m_rec_loss(logits)
            loss = NLL_loss

            precision = 1.0
            recall = 1.0

            if precision != 0 and recall != 0:
                loss_list.append(loss.item()) 
                precision_list.append(precision)
                recall_list.append(recall)

                tmp_loss_list.append(loss.item())
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

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
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
        with torch.no_grad():
            for pos_attr_set_batch, pos_attr_len_batch, neg_attr_set_batch, neg_attr_len_batch, negtarget_num_batch, user_batch, item_batch in eval_data:
                
                # print("pos_attr_set_batch", pos_attr_set_batch)

                pos_attr_set_gpu = pos_attr_set_batch.to(self.m_device)
                pos_attr_len_gpu = pos_attr_len_batch.to(self.m_device)

                neg_attr_set_gpu = neg_attr_set_batch.to(self.m_device)
                neg_attr_len_gpu = neg_attr_len_batch.to(self.m_device)
                negtarget_num_gpu = negtarget_num_batch.to(self.m_device)

                user_gpu = user_batch.to(self.m_device)

                item_gpu = item_batch.to(self.m_device)

                logits, mask_gpu = network.f_eval_forward(pos_attr_set_gpu, pos_attr_len_gpu, neg_attr_set_gpu, neg_attr_len_gpu, negtarget_num_gpu, user_gpu, item_gpu)

                NLL_loss = self.m_rec_loss(logits)
                loss = NLL_loss

                # print("pos_attr_set_gpu", pos_attr_set_gpu)

                mask_batch = ~mask_gpu.cpu()

                # print(~mask_batch)
                preds = self.f_greedy_decode(network, user_gpu, item_gpu)
                precision, recall, F1 = get_precision_recall_F1_set(preds.cpu(), pos_attr_set_batch, mask_batch, k=1)

                # precision = 1.0
                # recall = 1.0
                # F1 = 1.0

                loss_list.append(loss.item())
                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)

            logger_obj.f_add_output2IO("%d, precision:%.4f, recall:%.4f, F1:%.4f"%(self.m_eval_iteration, np.mean(precision_list), np.mean(recall_list), np.mean(F1_list)))
            logger_obj.f_add_output2IO("%d, loss:%.4f"%(self.m_eval_iteration, np.mean(loss_list)))

            logger_obj.f_add_scalar2tensorboard("eval/precision", np.mean(precision_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)

        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_precision = np.mean(precision_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        self.m_mean_eval_F1 =np.mean(F1_list)

        network.train()

    def f_greedy_decode(self, network, users, items):
        step_num = 1

        pre_attrs = None
        batch_size = users.size(0)
        voc_size = network.m_output_attr_embedding_user.weight.data.size(0)

        targets = torch.zeros(batch_size, voc_size).to(users.device)

        for i in range(step_num):
            # print(i)
            score = network.f_greedy_decode(users, items, pre_attrs, targets, i)
            
            # print("score", score.size())
            max_attr_index = torch.max(score, dim=1)[1]

            # print(max_attr_index)

            targets[torch.arange(batch_size), max_attr_index] = 1

            if pre_attrs is not None:
                new_pre_attrs = torch.cat([pre_attrs, max_attr_index.unsqueeze(-1)], dim=-1)
            else:
                new_pre_attrs = max_attr_index.unsqueeze(-1)

            pre_attrs = new_pre_attrs

        return pre_attrs
