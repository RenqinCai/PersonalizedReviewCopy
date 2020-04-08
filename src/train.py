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
from metric import Reconstruction_loss, KL_loss, RRe_loss, ARe_loss, KL_loss_z
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
        
        self.m_Recon_loss_fn = Reconstruction_loss(self.m_device, ignore_index=self.m_pad_idx)
        self.m_KL_loss_z_fn = KL_loss_z(self.m_device)
        self.m_KL_loss_s_fn = KL_loss(self.m_device)
        self.m_RRe_loss_fn = RRe_loss(self.m_device)
        self.m_ARe_loss_fn = ARe_loss(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_name = "REVIEWDI"

    # def saveModel(self, epoch, loss, recall, mrr):
    #     checkpoint = {
    #         'model': self.model.state_dict(),
    #         'args': self.args,
    #         'epoch': epoch,
    #         'optim': self.optim,
    #         'loss': loss,
    #         'recall': recall,
    #         'mrr': mrr
    #     }
        
    #     # checkpoint_dir = "../log/"+self.args.model_name+"/"+self.args.checkpoint_dir
    #     # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
    #     model_name = os.path.join(self.args.checkpoint_dir, "model_best.pt")
    #     # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
    #     torch.save(checkpoint, model_name)

    def f_save_model(self, epoch, network, optimizer):
        checkpoint = {'model':network.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer
        }

        now_time = datetime.datetime.now()
        time_name = str(now_time.day)+"_"+str(now_time.month)+"_"+str(now_time.hour)+"_"+str(now_time.minute)

        model_name = os.path.join(self.m_model_path, self.m_model_name+"/model_best_"+time_name+".pt")
        torch.save(checkpoint, model_name)

    def f_train(self, train_data, eval_data, network, optimizer):

        last_train_loss = 0
        last_eval_loss = 0

        for epoch in range(self.m_epochs):
            print("+"*20)

            s_time = datetime.datetime.now()
            self.f_train_epoch(train_data, network, optimizer, "train")
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
            print("-"*10, "validation dataset", "-"*10)
            self.f_train_epoch(eval_data, network, optimizer, "val")

            if last_eval_loss == 0:
                last_eval_loss = self.m_mean_val_loss
            elif last_eval_loss < self.m_mean_val_loss:
                print("!"*10, "overfitting validation loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
            else:
                
                print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_val_loss)
                last_eval_loss = self.m_mean_val_loss

    def f_train_epoch(self, data, network, optimizer, train_val_flag):

        train_loss_list = []
        eval_loss_list = []

        NLL_loss_list = []
        KL_loss_s_list = []
        KL_loss_z_list = []
        RRe_loss_list = []
        ARe_loss_list = []

        # (NLL_loss+KL_weight_s*KL_loss_s+KL_weight_z*KL_loss_z+RRe_loss+ARe_loss)

        batch_size = self.m_batch_size

        for input_batch, user_batch, target_batch, ARe_batch, RRe_batch, length_batch in data:

            if train_val_flag == "train":
                network.train()
                self.m_train_step += 1
            elif train_val_flag == "val":
                network.eval()
                # self.m_valid_step += 1
                eval_flag = random.randint(1,101)
                if eval_flag != 10:
                    continue
            else:
                raise NotImplementedError

            input_batch = input_batch.to(self.m_device)
            user_batch = user_batch.to(self.m_device)
            length_batch = length_batch.to(self.m_device)
            target_batch = target_batch.to(self.m_device)
            RRe_batch = RRe_batch.to(self.m_device)
            ARe_batch = ARe_batch.to(self.m_device)

            logp, z_mean_prior, z_mean, z_logv, z, s_mean, s_logv, s, ARe_pred, RRe_pred = network(input_batch, user_batch, length_batch)

            ### NLL loss
            NLL_loss = self.m_Recon_loss_fn(logp, target_batch, length_batch)

            ### KL loss
            KL_loss_z, KL_weight_z = self.m_KL_loss_z_fn(z_mean_prior, z_mean, z_logv, self.m_train_step, self.m_k, self.m_x0, self.m_anneal_func)

            KL_loss_s, KL_weight_s = self.m_KL_loss_s_fn(s_mean, s_logv, self.m_train_step, self.m_k, self.m_x0, self.m_anneal_func)

            ### RRe loss
            RRe_loss = self.m_RRe_loss_fn(RRe_pred, RRe_batch)

            ### ARe loss
            ARe_loss = self.m_ARe_loss_fn(ARe_pred, ARe_batch)

            loss = (NLL_loss+KL_weight_s*KL_loss_s+KL_weight_z*KL_loss_z+RRe_loss+ARe_loss)/batch_size

            # print("reconstruction loss:%.4f"%NLL_loss.item(), "\t KL loss z:%.4f"%KL_loss_z.item(), "\t KL loss s%.4f"%KL_loss_s.item(), "\tRRe loss:%.4f"%RRe_loss.item(), "\t ARe loss:%.4f"%ARe_loss.item())
            
            if train_val_flag == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
            elif train_val_flag == "val":
                eval_loss_list.append(loss.item())

            NLL_loss_list.append(NLL_loss.item())
            KL_loss_z_list.append(KL_loss_z.item())
            KL_loss_s_list.append(KL_loss_s.item())
            RRe_loss_list.append(RRe_loss.item())
            ARe_loss_list.append(ARe_loss.item())
        
        print("avg nll loss:%.4f"%np.mean(NLL_loss_list), end=', ')
        print("avg kl loss z:%.4f"%np.mean(KL_loss_z_list), end=', ')
        print("avg kl loss s:%.4f"% np.mean(KL_loss_s_list), end=', ')
        print("RRe loss:%.4f"% np.mean(RRe_loss_list), end=', ')
        print("ARe loss:%.4f"% np.mean(ARe_loss_list))

        if train_val_flag == "train":
            self.m_mean_train_loss = np.mean(train_loss_list)
        elif train_val_flag == "val":
            self.m_mean_val_loss = np.mean(eval_loss_list)

            
        


