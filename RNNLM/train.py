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
import random

from metric import Reconstruction_loss, KL_loss, RRe_loss, ARe_loss, KL_loss_z
from model import REVIEWDI
from inference import INFER
from data_ref import get_batches


class TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_pad_idx = vocab_obj.pad_idx

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
        # self.m_seq_size = seq_size

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func
        
        self.m_Recon_loss_fn = Reconstruction_loss(ignore_index=self.m_pad_idx, device=self.m_device)
        self.m_KL_loss_z_fn = KL_loss_z(self.m_device)
        self.m_KL_loss_s_fn = KL_loss(self.m_device)
        self.m_RRe_loss_fn = RRe_loss(self.m_device)
        self.m_ARe_loss_fn = ARe_loss(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        # self.m_model_path = args.model_path
        self.m_model_name = "RNNLM"

        self.m_train_iteration = 0
        self.m_valid_iteration = 0

        self.m_KL_weight = 0.0
        self.m_model_file = args.model_file

    def f_save_model(self, epoch, network, optimizer):
        checkpoint = {'model':network.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer
        }

        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, valid_data, network, optimizer, logger_obj):

        last_train_loss = 0
        last_eval_loss = 0

        self.m_train_iteration = 0
        self.m_valid_iteration = 0

        for epoch in range(self.m_epochs):
            print("+"*20)
            
            s_time = datetime.datetime.now()

            self.f_train_epoch(train_data, network, optimizer, logger_obj, "train")
            e_time = datetime.datetime.now()
            print("epoch train duration", e_time-s_time)
            
            logger_obj.f_add_scalar2tensorboard("train/loss", self.m_mean_train_loss.item(), epoch)
            if last_train_loss == 0:
                last_train_loss = self.m_mean_train_loss

            elif last_train_loss < self.m_mean_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                last_train_loss = self.m_mean_train_loss

            self.f_save_model(epoch, network, optimizer)

            s_time = datetime.datetime.now()
            self.f_train_epoch(valid_data, network, optimizer, logger_obj, "val")
            e_time = datetime.datetime.now()
            print("epoch validate duration", e_time-s_time)

            if last_eval_loss == 0:
                last_eval_loss = self.m_mean_val_loss
            elif last_eval_loss < self.m_mean_val_loss:
                print("!"*10, "overfitting validation loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
            else:
                
                print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
                last_eval_loss = self.m_mean_val_loss

    def f_train_epoch(self, data, network, optimizer, logger_obj, train_val_flag):

        train_loss_list = []
        eval_loss_list = []

        NLL_loss_list = []
        KL_loss_s_list = []
        KL_loss_z_list = []
        RRe_loss_list = []
        ARe_loss_list = []

        batch_size = self.m_batch_size
        # seq_size = self.m_seq_size
        # batches = get_batches(in_text, out_text, batch_size, seq_size)
        # for x, y, length in batches:
        # for input_batch, target_batch, length_batch in batches:
        for input_batch, target_batch, length_batch in data: 
            # input_batch = torch.tensor(input_batch)
            # target_batch = torch.tensor(target_batch)
            # length_batch = torch.tensor(length_batch)
            

            if train_val_flag == "train":
                network.train()
                self.m_train_step += 1
            elif train_val_flag == "val":
                network.eval()
                eval_flag = random.randint(1,10)
                if eval_flag != 10:
                    continue
            else:
                raise NotImplementedError

            input_batch = input_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)

            # print("ok")
            # exit()
            logp = network(input_batch, length_batch)
            # exit()
            ### NLL loss
            NLL_loss = self.m_Recon_loss_fn(logp, target_batch, length_batch)
            # print("length batch", length_batch.size())
            NLL_loss = NLL_loss/sum(length_batch)
            # NLL_loss = NLL_loss/batch_size
            # exit()

            ### KL loss
            # KL_loss_z, KL_weight_z = self.m_KL_loss_z_fn(z_mean_prior, z_mean, z_logv, self.m_step, self.m_k, self.m_x0, self.m_anneal_func)

            # KL_loss_z, KL_weight_z = self.m_KL_loss_s_fn(z_mean, z_logv, self.m_train_step, self.m_k, self.m_x0, self.m_anneal_func)
            # KL_loss_z = KL_loss_z/batch_size

            # self.m_KL_weight = KL_weight_z
            # ### RRe loss
            # RRe_loss = self.m_RRe_loss_fn(RRe_pred, RRe_batch)

            # ### ARe loss
            # ARe_loss = self.m_ARe_loss_fn(ARe_pred, ARe_batch)
            loss = NLL_loss
            # loss = (NLL_loss+KL_weight_z*KL_loss_z)

            if train_val_flag == "train":
                # logger_obj.f_add_scalar2tensorboard("train/KL_loss", KL_loss_z.item(), self.m_train_iteration)
                # logger_obj.f_add_scalar2tensorboard("train/KL_weight", KL_weight_z, self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/loss", loss.item(), self.m_train_iteration)
                # logger_obj.f_add_scalar2tensorboard("train/NLL_loss", NLL_loss.item(), self.m_train_iteration)
            else:
                # logger_obj.f_add_scalar2tensorboard("valid/KL_loss", KL_loss_z.item(), self.m_valid_iteration)
                # logger_obj.f_add_scalar2tensorboard("valid/KL_weight", KL_weight_z, self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/loss", loss.item(), self.m_valid_iteration)
                # logger_obj.f_add_scalar2tensorboard("valid/NLL_loss", NLL_loss.item(), self.m_valid_iteration)
            
            if train_val_flag == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                self.m_train_iteration += 1

                _ = torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                if self.m_train_iteration %1000 == 0:
                    print(self.m_train_iteration, " training batch")
                    print("--"*10)
            elif train_val_flag == "val":
                self.m_valid_iteration += 1
                eval_loss_list.append(loss.item())

        if train_val_flag == "train":
            self.m_mean_train_loss = np.mean(train_loss_list)
        elif train_val_flag == "val":
            self.m_mean_val_loss = np.mean(eval_loss_list)

            
        


