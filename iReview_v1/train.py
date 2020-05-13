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
from metric import _REC_LOSS, _KL_LOSS_CUSTOMIZE, _KL_LOSS_STANDARD, _RRE_LOSS, _ARE_LOSS
from model import REVIEWDI
from inference import INFER
import random

class TRAINER(object):

    def __init__(self, vocab, args, device):
        super().__init__()

        self.m_pad_idx = vocab.pad_idx

        self.m_Recon_loss_fn = None
        self.m_KL_loss_fn = None
        self.m_RRe_loss_fn = None
        self.m_ARe_loss_fn = None

        self.m_save_mode = True
        self.m_mean_train_loss = 0
        self.m_mean_val_loss = 0
        
        self.m_device = device

        self.m_epochs = args.epochs
        self.m_batch_size = args.batch_size

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func
        
        self.m_rec_loss = _REC_LOSS(self.m_device, ignore_index=self.m_pad_idx)
        self.m_kl_loss_z = _KL_LOSS_STANDARD(self.m_device)
        self.m_kl_loss_s = _KL_LOSS_STANDARD(self.m_device)
        self.m_kl_loss_l = _KL_LOSS_STANDARD(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_print_interval = args.print_interval

    def f_save_model(self, epoch, network, optimizer):
        checkpoint = {'model':network.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer
        }

        # now_time = datetime.datetime.now()
        # time_name = str(now_time.day)+"_"+str(now_time.month)+"_"+str(now_time.hour)+"_"+str(now_time.minute)

        # model_name = os.path.join(self.m_model_path, self.m_model_name+"/model_best_"+time_name+".pt")
        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, eval_data, network, optimizer, logger_obj):

        last_train_loss = 0
        last_eval_loss = 0

        self.m_train_iteration = 0
        self.m_valid_iteration = 0

        for epoch in range(self.m_epochs):
            print("++"*20, epoch, "++"*20)

            s_time = datetime.datetime.now()
            self.f_train_epoch(train_data, network, optimizer, logger_obj, "train")
            e_time = datetime.datetime.now()

            print("epoch duration", e_time-s_time)

            if last_train_loss == 0:
                last_train_loss = self.m_mean_train_loss

            elif last_train_loss < self.m_mean_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                last_train_loss = self.m_mean_train_loss

            self.f_save_model(epoch, network, optimizer)
            # self.m_infer.f_inference_epoch()
            # print("-"*10, "validation dataset", "-"*10)
            # self.f_train_epoch(eval_data, network, optimizer, logger_obj, "val")

            # if last_eval_loss == 0:
            #     last_eval_loss = self.m_mean_val_loss
            # elif last_eval_loss < self.m_mean_val_loss:
            #     print("!"*10, "overfitting validation loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
            # else:
                
            #     print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
            #     last_eval_loss = self.m_mean_val_loss

    def f_get_KL_weight(self, anneal_func="beta"):
        weight = 0
        if anneal_func == "beta":
            weight = 0.1
        elif anneal_func == "logistic":
            k = 0.001
            x0 = 2000
            weight = float(1/(1+np.exp(-k*(step-x0))))
        else:
            raise NotImplementedError

        return weight

    def f_train_epoch(self, data, network, optimizer, logger_obj, train_val_flag):

        train_loss_list = []
        eval_loss_list = []

        loss_list = []
        NLL_loss_list = []
        KL_loss_list = []

        # batch_size = self.m_batch_size
        iteration = 0
        if train_val_flag == "train":
            network.train()
        elif train_val_flag == "val":
            network.eval()

        for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch in data:

            if train_val_flag == "train":
                self.m_train_step += 1
            elif train_val_flag == "val":
                # self.m_valid_step += 1
                eval_flag = random.randint(1,101)
                if eval_flag != 10:
                    continue
            else:
                raise NotImplementedError

            input_batch = input_batch.to(self.m_device)
            input_length_batch = input_length_batch.to(self.m_device)

            user_batch = user_batch.to(self.m_device)
            item_batch = item_batch.to(self.m_device)
            
            target_batch = target_batch.to(self.m_device)
            target_length_batch = target_length_batch.to(self.m_device)

            input_de_batch = target_batch
            input_de_length_batch = target_length_batch

            logits, z_mean, z_logv, z, s_mean, s_logv, s, l_mean, l_logv, l, variational_hidden = network(input_batch, input_length_batch, input_de_batch, input_de_length_batch, user_batch, random_flag)

            batch_size = input_batch.size(0)
            ### NLL loss
            NLL_loss = self.m_rec_loss(logits, target_batch, target_length_batch)
            NLL_loss = NLL_loss/batch_size

            loss = NLL_loss
            KL_loss = None
            if random_flag == 0:
                KL_loss_z = self.m_kl_loss_z(z_mean, z_logv)
                KL_weight_z = self.f_get_KL_weight()
                KL_loss_z = KL_loss_z/batch_size

                KL_loss_s = self.m_kl_loss_s(s_mean, s_logv)
                KL_weight_s = self.f_get_KL_weight()
                KL_loss_s = KL_loss_s/batch_size

                KL_loss_l = self.m_kl_loss_l(l_mean, l_logv)
                KL_weight_l = self.f_get_KL_weight()
                KL_loss_l = KL_loss_l/batch_size

                KL_loss = KL_weight_z*KL_loss_z+KL_weight_s*KL_loss_s+KL_weight_l*KL_loss_l
                loss = loss + KL_loss
            elif random_flag == 1:
                KL_loss_z = self.m_kl_loss_z(z_mean, z_logv)
                KL_weight_z = self.f_get_KL_weight()
                KL_loss_z = KL_loss_z/batch_size

                KL_loss_s = self.m_kl_loss_s(s_mean, s_logv)
                KL_weight_s = self.f_get_KL_weight()
                KL_loss_s = KL_loss_s/batch_size

                KL_loss = KL_weight_z*KL_loss_z+KL_weight_s*KL_loss_s
                loss = loss + KL_loss
            elif random_flag == 2:

                KL_loss_s = self.m_kl_loss_s(s_mean, s_logv)
                KL_weight_s = self.f_get_KL_weight()
                KL_loss_s = KL_loss_s/batch_size

                KL_loss_l = self.m_kl_loss_l(l_mean, l_logv)
                KL_weight_l = self.f_get_KL_weight()
                KL_loss_l = KL_loss_l/batch_size

                KL_loss = KL_weight_s*KL_loss_s+KL_weight_l*KL_loss_l
                loss = loss + KL_loss
            elif random_flag == 3:
                KL_loss_z = self.m_kl_loss_z(z_mean, z_logv)
                KL_weight_z = self.f_get_KL_weight()
                KL_loss_z = KL_loss_z/batch_size

                KL_loss_l = self.m_kl_loss_l(l_mean, l_logv)
                KL_weight_l = self.f_get_KL_weight()
                KL_loss_l = KL_loss_l/batch_size

                KL_loss = KL_weight_z*KL_loss_z+KL_weight_l*KL_loss_l
                loss = loss + KL_loss
            else:
                raise NotImplementedError("0, 1, 2, 3, variational not defined!")
            
            # print("reconstruction loss:%.4f"%NLL_loss.item(), "\t KL loss z:%.4f"%KL_loss_z.item(), "\t KL loss s%.4f"%KL_loss_s.item(), "\tRRe loss:%.4f"%RRe_loss.item(), "\t ARe loss:%.4f"%ARe_loss.item())
            
            if train_val_flag == "train":
                logger_obj.f_add_scalar2tensorboard("train/KL_loss", KL_loss.item(), self.m_train_iteration)
                # logger_obj.f_add_scalar2tensorboard("train/KL_weight_z", KL_weight_z, self.m_train_iteration)
                # logger_obj.f_add_scalar2tensorboard("train/KL_loss_s", KL_loss_s.item(), self.m_train_iteration)
                # logger_obj.f_add_scalar2tensorboard("train/KL_weight_s", KL_weight_s, self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/loss", loss.item(), self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/NLL_loss", NLL_loss.item(), self.m_train_iteration)
                
            else:
                logger_obj.f_add_scalar2tensorboard("valid/KL_loss", KL_loss_z.item(), self.m_valid_iteration)
                # logger_obj.f_add_scalar2tensorboard("valid/KL_weight", KL_weight_z, self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/loss", loss.item(), self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/NLL_loss", NLL_loss.item(), self.m_valid_iteration)
                # logger_obj.f_add_scalar2tensorboard("valid/ARe_loss", ARe_loss.item(), self.m_valid_iteration)
                # logger_obj.f_add_scalar2tensorboard("valid/RRe_loss", RRe_loss.item(), self.m_valid_iteration)

            if train_val_flag == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                self.m_train_iteration += 1
        
            elif train_val_flag == "val":
                eval_loss_list.append(loss.item())
                self.m_valid_iteration += 1

            loss_list.append(loss.item())
            NLL_loss_list.append(NLL_loss.item())
            KL_loss_list.append(KL_loss.item())

            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%04d, Loss %9.4f, NLL-Loss %9.4f, KL Loss %.4f"
                    %(iteration, np.mean(loss_list), np.mean(NLL_loss_list), np.mean(KL_loss_list)))

                loss_list = []
                NLL_loss_list = []
                KL_loss_list = []

        if train_val_flag == "train":
            self.m_mean_train_loss = np.mean(train_loss_list)
        elif train_val_flag == "val":
            self.m_mean_val_loss = np.mean(eval_loss_list)

            
        


