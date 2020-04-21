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

from metric import Reconstruction_loss, KL_loss_standard, KL_loss_customize
from model import REVIEWDI
from inference import INFER

class TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_pad_idx = vocab_obj.pad_idx

        self.m_Recon_loss_fn = None
        self.m_KL_loss_fn = None

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
        
        self.m_Recon_loss_fn = Reconstruction_loss(ignore_index=self.m_pad_idx, device=self.m_device)
        self.m_KL_loss_z_fn = KL_loss_standard(self.m_device, self.m_anneal_func)
    
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

    def f_save_grad(self, name):
        def hook(grad):
            # if name not in self.m_grads:
            self.m_grads[name] = grad
            # else:
            #     self.m_grads[name] += grad
        return hook

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
            print("+"*20)
            
            s_time = datetime.datetime.now()
            self.f_train_epoch(train_data, network, optimizer, logger_obj, "train")
            e_time = datetime.datetime.now()
            print("epoch train duration", e_time-s_time)
            
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
        for input_batch, target_batch, length_batch in data:
            
            if train_val_flag == "train":
                network.train()
                self.m_train_step += 1
            elif train_val_flag == "val":
                network.eval()
                eval_flag = random.randint(1,5)
                if eval_flag != 1:
                    continue
            else:
                raise NotImplementedError

            input_batch = input_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
        
            logp, z_mean, z_logv, z, init_de_hidden, last_en_hidden = network(input_batch, length_batch)

            # z_mean.register_hook(self.f_save_grad('z_mean'))
            # z.register_hook(self.f_save_grad('z'))

            init_de_hidden.register_hook(self.f_save_grad('init_de_hidden'))
            last_en_hidden.register_hook(self.f_save_grad('last_en_hidden'))

            ### NLL loss
            NLL_loss = self.m_Recon_loss_fn(logp, target_batch, length_batch)
            # NLL_loss = NLL_loss/torch.sum(length_batch)
            NLL_loss = NLL_loss/batch_size

            KL_loss_z, KL_weight_z = self.m_KL_loss_z_fn(z_mean, z_logv, self.m_train_step)

            # KL_loss_z, KL_weight_z = self.m_KL_loss_z_fn(z_mean, z_logv, self.m_train_step, self.m_k, self.m_x0, anneal_func=self.m_anneal_func)
            KL_loss_z = KL_loss_z/batch_size
            # KL_loss_z = KL_loss_z/torch.sum(length_batch)

            self.m_KL_weight = KL_weight_z
           
            loss = (NLL_loss+KL_weight_z*KL_loss_z)

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

                self.m_train_iteration += 1
                if self.m_train_iteration %4000 == 0:
                    print(self.m_train_iteration, " training batch")
                    print("--"*10)

                    print("init_de_hidden", self.m_grads['init_de_hidden'])
                    print("last_en_hidden", self.m_grads['last_en_hidden'])
                    print("z", z)
                    print("encoder gradient", network.m_encoder_rnn.weight_ih_l0.grad)
                    print("m_hidden2mean_z.weight", network.m_hidden2mean_z.weight.grad)
                    print("m_latent2hidden.weight", network.m_latent2hidden.weight.grad)
                    print("decoder gradient", network.m_decoder_rnn.weight_ih_l0.grad)

                optimizer.step()
                train_loss_list.append(loss.item())

                # _ = torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)

            iteration += 1 
            if iteration % 200 == 0:
                print("%04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                    %(iteration, loss.item(), NLL_loss.item(), KL_loss_z.item(), KL_weight_z))

                # if self.m_train_iteration > 0:
                #     exit()

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

            
        


