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
from loss import _BPR_LOSS
from metric import get_precision_recall
from model import _ATTR_NETWORK
# from infer_new import _INFER
import random

class _TRAINER(object):

    def __init__(self, args, device):
        super().__init__()

        self.m_device = device

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_train_precision = 0
        self.m_mean_train_recall = 0

        self.m_mean_val_loss = 0
        self.m_mean_eval_precision = 0
        self.m_mean_eval_recall = 0
        
        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        self.m_rec_loss = _BPR_LOSS(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        # self.m_l2_reg = args.l2_reg

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

    def f_train(self, train_data, eval_data, network, optimizer, logger_obj, local_rank):
        last_train_loss = 0
        last_eval_loss = 0

        overfit_indicator = 0

        best_eval_precision = 0
        # self.f_init_word_embed(pretrain_word_embed, network)
        try: 
            for epoch in range(self.m_epochs):
                
                print("++"*10, epoch, "++"*10)

                s_time = datetime.datetime.now()
                self.f_eval_epoch(eval_data, network, optimizer, logger_obj, local_rank)
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

                if local_rank is not None:
                    if local_rank == 0:
                        if best_eval_precision < self.m_mean_eval_precision:
                            checkpoint = {'model':network.module.state_dict()}
                            self.f_save_model(checkpoint)
                            best_eval_precision = self.m_mean_eval_precision
                else:
                    if best_eval_precision < self.m_mean_eval_precision:
                            checkpoint = {'model':network.state_dict()}
                            self.f_save_model(checkpoint)
                            best_eval_precision = self.m_mean_eval_precision

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
            if local_rank is not None:
                if local_rank == 0:
                    if best_eval_precision < self.m_mean_eval_precision:
                        print("... final save ...")
                        checkpoint = {'model':network.module.state_dict()}
                        self.f_save_model(checkpoint)
                        best_eval_precision = self.m_mean_eval_precision
                exit()
            else:
                if best_eval_precision < self.m_mean_eval_precision:
                        print("... final save ...")
                        checkpoint = {'model':network.state_dict()}
                        self.f_save_model(checkpoint)
                        best_eval_precision = self.m_mean_eval_precision
            
    def f_get_pred(self, network, user_ids, item_ids, local_rank):
        if local_rank is not None:
            ### user_x = batch_size*embed_size
            user_x = network.module.m_user_embedding(user_ids)

            ### item_x = batch_size*embed_size
            item_x = network.module.m_item_embedding(item_ids)

            ### m_tag_user_embedding: tag_num*embed_size
            ### user_logits: batch_size*tag_num
            user_logits = torch.matmul(user_x, network.module.m_tag_user_embedding.weight.data.transpose(0, 1))
            # user_logits = network.m_tag_user_embedding.weight.data*user_x

            ### m_tag_item_embedding: tag_num*embed_size
            ### item_logits: batch_size*tag_num
            item_logits = torch.matmul(item_x, network.module.m_tag_item_embedding.weight.data.transpose(0, 1))
            # item_logits = network.m_tag_item_embedding.weight.data*item_x

        else:
            user_x = network.m_user_embedding(user_ids)
            item_x = network.m_item_embedding(item_ids)
            user_logits = torch.matmul(user_x, network.m_tag_user_embedding.weight.data.transpose(0, 1))
            item_logits = torch.matmul(item_x, network.m_tag_item_embedding.weight.data.transpose(0, 1))
        
        ### pred_logits: batch_size*tag_num
        pred_logits = user_logits+item_logits

        return pred_logits

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_list = []
        precision_list = []
        recall_list = []

        iteration = 0

        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)

        tmp_loss_list = []
        tmp_precision_list = []
        tmp_recall_list = []

        network.train()
        for pos_tag_batch, neg_tag_batch, user_batch, item_batch in train_data:
            pos_tag_batch_gpu = pos_tag_batch.to(self.m_device)
            neg_tag_batch_gpu = neg_tag_batch.to(self.m_device)

            user_batch_gpu = user_batch.to(self.m_device)
            item_batch_gpu = item_batch.to(self.m_device)

            logits, user_x, item_x, pos_user_tag_x, pos_item_tag_x, neg_user_tag_x, neg_item_tag_x = network(pos_tag_batch_gpu, neg_tag_batch_gpu, user_batch_gpu, item_batch_gpu)
            # print("logits", logits)
            # if torch.isnan(logits).any():
            #     print("logits nan")
            #     exit()

            NLL_loss = self.m_rec_loss(logits)
            # if torch.isnan(NLL_loss).any():
            #     print("NLL_loss nan")
            #     exit()
            # print("NLL_loss", NLL_loss.item())
            
            # reg_loss = None
            # for w in network.parameters():
            #     if reg_loss is None:
            #         reg_loss = w.norm(2)
            #     else:
            #         reg_loss += w.norm(2)
            
            # reg_loss = user_x.norm(2)+item_x.norm(2)+pos_user_tag_x.norm(2)+pos_item_tag_x.norm(2)+neg_user_tag_x.norm(2)+neg_item_tag_x.norm(2)

            # loss = NLL_loss+self.m_l2_reg*reg_loss
            loss = NLL_loss

            # preds = self.f_get_pred(network, user_batch_gpu, item_batch_gpu)

            # precision, recall = get_precision_recall(preds.cpu(), pos_tag_batch, k=3)
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

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj, local_rank):
        loss_list = []
        precision_list = []
        recall_list = []

        iteration = 0
        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        network.eval()
        with torch.no_grad():
            for pos_tag_batch, mask_batch, user_batch, item_batch in eval_data:

                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue


                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                # pos_tag_batch_gpu = pos_tag_batch.to(self.m_device)

                # logits = network(pos_tag_batch_gpu, , user_batch_gpu, item_batch_gpu)

                # NLL_loss = self.m_rec_loss(logits)
                
                # loss = NLL_loss
                loss = 0.0

                preds = self.f_get_pred(network, user_batch_gpu, item_batch_gpu, local_rank)
                precision, recall = get_precision_recall(preds.cpu(), pos_tag_batch, mask_batch, k=3)

                if precision != 0 and recall != 0:
                    # loss_list.append(loss.item()) 
                    loss_list.append(loss)
                    precision_list.append(precision)
                    recall_list.append(recall)

            logger_obj.f_add_output2IO("%d, NLL_loss:%.4f, precision:%.4f, recall:%.4f"%(self.m_eval_iteration, np.mean(loss_list), np.mean(precision_list), np.mean(recall_list)))

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/precision", np.mean(precision_list), self.m_eval_iteration)
            logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)
                
        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_precision = np.mean(precision_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        network.train()

