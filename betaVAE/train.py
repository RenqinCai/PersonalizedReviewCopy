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

from metric import _REC_LOSS, _KL_LOSS_STANDARD, _KL_LOSS_CUSTOMIZE
from model import _NETWORK
from inference import _INFER

class _TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_pad_idx = vocab_obj.pad_idx

        self.m_save_mode = True
        self.m_mean_train_loss = 0
        self.m_mean_val_loss = 0
        
        self.m_device = device

        self.m_epochs = args.epochs
        self.m_batch_size = args.batch_size

        self.m_x0 = args.x0
        self.m_k = args.k

        self.m_anneal_func = args.anneal_func
        print("args.anneal_func", args.anneal_func)
        
        self.m_Recon_loss_fn = _REC_LOSS(ignore_index=self.m_pad_idx, device=self.m_device)
        self.m_KL_loss_z_fn = _KL_LOSS_STANDARD(self.m_device)
    
        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_name = "VAE"

        self.m_train_iteration = 0
        self.m_valid_iteration = 0

        self.m_KL_weight = 0.0
        # self.m_KL_weight_list = []

        self.m_grads = {}
        self.m_model_file = args.model_file

        self.m_train_loss_list = []
        self.m_train_rec_loss_list = []
        self.m_train_kl_loss_list = []

        self.m_print_interval = args.print_interval

    def f_save_grad(self, name):
        def hook(grad):
            # if name not in self.m_grads:
            self.m_grads[name] = grad
            # else:
            #     self.m_grads[name] += grad
        return hook
    
    def f_get_KL_weight(self, step):
        weight = 0
        if self.m_anneal_func == "beta":
            weight = 0.1
        elif self.m_anneal_func == "logistic":
            k = 0.001
            x0 = 2000
            weight = float(1/(1+np.exp(-k*(step-x0))))
        else:
            raise NotImplementedError

        return weight

    def f_save_model(self, epoch, network, optimizer):
        checkpoint = {'model':network.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer
        }

        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, eval_data, network, optimizer, logger_obj):

        last_train_loss = 0
        last_eval_loss = 0

        self.m_train_iteration = 0
        self.m_valid_iteration = 0

        for epoch in range(self.m_epochs):
            print("+"*20)
            
            s_time = datetime.datetime.now()
            self.f_train_epoch(train_data, network, optimizer, logger_obj, "train")
            e_time = datetime.datetime.now()
            print(epoch, "epoch train duration", e_time-s_time)
            
            if last_train_loss == 0:
                last_train_loss = self.m_mean_train_loss

            elif last_train_loss < self.m_mean_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_train_loss)
                last_train_loss = self.m_mean_train_loss

            self.f_save_model(epoch, network, optimizer)
            # self.m_infer.f_inference_epoch()
            s_time = datetime.datetime.now()
            self.f_train_epoch(eval_data, network, optimizer, logger_obj, "val")
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
        iteration = 0
        if train_val_flag == "train":
            network.train()
            print("++"*10, "training", "++"*10)
        elif train_val_flag == "val":
            network.eval()
            print("++"*10, "val", "++"*10)
        else:
            raise NotImplementedError
        
        for input_batch, target_batch, length_batch, _, _ in data:
            
            if train_val_flag == "train":
                self.m_train_step += 1
                
            elif train_val_flag == "val":
                eval_flag = random.randint(1,5)
                if eval_flag != 1:
                    continue
            else:
                raise NotImplementedError

            input_batch = input_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
        
            logp, z_mean, z_logv, z, last_en_hidden, init_de_hidden = network(input_batch, length_batch)

            ### NLL loss
            NLL_loss = self.m_Recon_loss_fn(logp, target_batch, length_batch)
            
            NLL_loss = NLL_loss/len(z)
            
            KL_weight_z = self.f_get_KL_weight(0)
            KL_loss_z = self.m_KL_loss_z_fn(z_mean, z_logv)

            KL_loss_z = KL_loss_z/len(z)
            
            self.m_KL_weight = KL_weight_z
           
            loss = (NLL_loss+KL_weight_z*KL_loss_z)
            # exit()
            if train_val_flag == "train":
                logger_obj.f_add_scalar2tensorboard("train/KL_loss", KL_loss_z.item(), self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/KL_weight", KL_weight_z, self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/loss", loss.item(), self.m_train_iteration)
                logger_obj.f_add_scalar2tensorboard("train/NLL_loss", NLL_loss.item(), self.m_train_iteration)
            else:
                logger_obj.f_add_scalar2tensorboard("valid/KL_loss", KL_loss_z.item(), self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/KL_weight", KL_weight_z, self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/loss", loss.item(), self.m_valid_iteration)
                logger_obj.f_add_scalar2tensorboard("valid/NLL_loss", NLL_loss.item(), self.m_valid_iteration)
            
            if train_val_flag == "train":
                optimizer.zero_grad()
                loss.backward()

                self.m_train_loss_list.append(loss.item())
                self.m_train_rec_loss_list.append(NLL_loss.item())
                self.m_train_kl_loss_list.append(KL_loss_z.item())
                self.m_train_iteration += 1
    
                optimizer.step()
                train_loss_list.append(loss.item())

                # _ = torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)

            iteration += 1 
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                    %(iteration, np.mean(self.m_train_loss_list), np.mean(self.m_train_rec_loss_list), np.mean(self.m_train_kl_loss_list), KL_weight_z))

                self.m_train_loss_list = []
                self.m_train_rec_loss_list = []
                self.m_train_kl_loss_list = []

            elif train_val_flag == "val":
                self.m_valid_iteration += 1
                eval_loss_list.append(loss.item())

            NLL_loss_list.append(NLL_loss.item())
            KL_loss_z_list.append(KL_loss_z.item())
 
        print("avg kl loss z:%.4f"%np.mean(KL_loss_z_list), end=', ')
        print("KL weight:%.4f"%self.m_KL_weight)
        
        if train_val_flag == "train":
            self.m_mean_train_loss = np.mean(train_loss_list)
        elif train_val_flag == "val":
            self.m_mean_val_loss = np.mean(eval_loss_list)

            
        


