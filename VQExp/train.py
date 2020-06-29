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
from model import _NETWORK
from infer_new import _INFER
import random

class _TRAINER(object):

    def __init__(self, vocab, args, device):
        super().__init__()

        self.m_pad_idx = vocab.pad_idx

        self.m_Recon_loss_fn = None
        self.m_KL_loss_fn = None
        self.m_RRe_loss_fn = None
        self.m_ARe_loss_fn = None

        self.m_save_mode = True

        self.m_mean_pre_train_loss = 0
        self.m_mean_pre_val_loss = 0

        self.m_mean_en_train_loss = 0
        self.m_mean_en_val_loss = 0

        self.m_mean_de_train_loss = 0
        self.m_mean_de_val_loss = 0
        
        self.m_device = device

        self.m_epochs = args.epochs
        self.m_batch_size = args.batch_size

        self.m_x0 = args.x0
        self.m_k = args.k
        self.m_decay = 0.5

        self.m_anneal_func = args.anneal_func
        
        self.m_rec_loss = _REC_LOSS(self.m_device, ignore_index=self.m_pad_idx)
        self.m_kl_loss_z = _KL_LOSS_CUSTOMIZE(self.m_device)
        self.m_kl_loss_s = _KL_LOSS_CUSTOMIZE(self.m_device)
        self.m_kl_loss_l = _KL_LOSS_STANDARD(self.m_device)

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_print_interval = args.print_interval
        self.m_overfit_epoch_threshold = 3

    def f_save_model(self, epoch, network, en_optimizer, de_optimizer):
        checkpoint = {'model':network.state_dict(),
            'epoch': epoch,
            'en_optimizer': en_optimizer,
            'de_optimizer': de_optimizer
        }

        # now_time = datetime.datetime.now()
        # time_name = str(now_time.day)+"_"+str(now_time.month)+"_"+str(now_time.hour)+"_"+str(now_time.minute)

        # model_name = os.path.join(self.m_model_path, self.m_model_name+"/model_best_"+time_name+".pt")
        torch.save(checkpoint, self.m_model_file)

    def f_initialize_network(self, train_data, pretrain_newtork, network):
        batch_index = 0

        user_embedding_shape = network.m_user_embedding.size()

        latent_size = user_embedding_shape[0]
        cluster_num = user_embedding_shape[1]
        
        pretrain_newtork.eval()

        network.m_embedding.weight.data.copy_(pretrain_newtork.m_embedding.weight.data)

        # global_user_embedding = torch.zeros(latent_size).to(self.m_device)
        # cluster_update_batch_interval = 100

        cluster_index = 0
        for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in train_data:
            input_batch_gpu = input_batch.to(self.m_device)
            input_length_batch_gpu = input_length_batch.to(self.m_device)

            user_hidden = pretrain_newtork.m_user_encoder(input_batch_gpu, input_length_batch_gpu)
            
            # avg_user_embedding = torch.mean(user_hidden, dim=0)
            
            # # if batch_index % cluster_update_batch_interval == 0:
            # cluster_index = int(batch_index/cluster_update_batch_interval)

            batch_size = input_batch.size(0)
            
            eval_flag = random.randint(1,101)
            if eval_flag != 10:
                random_sample_index = random.randint(0, batch_size-1)
                network.m_user_embedding.data[:, cluster_index].add_(user_hidden[random_sample_index])
                cluster_index += 1

            if cluster_index >= cluster_num:
                break

            # network.m_user_embedding.data[:, cluster_index].add_(avg_user_embedding)
            
            # batch_index += 1

            # print("avg_user_embedding", avg_user_embedding)
            # global_user_embedding = global_user_embedding + avg_user_embedding

        # network.m_user_embedding.data.div_(cluster_update_batch_interval)

        user_encoder_dict = network.m_user_item_encoder.m_user_encoder.state_dict()
        pretrained_user_encoder_dict = pretrain_newtork.m_user_encoder.state_dict()

        common_state_dict = {k:v for k, v in pretrained_user_encoder_dict.items() if k in user_encoder_dict}

        user_encoder_dict.update(common_state_dict)
        network.m_user_item_encoder.m_user_encoder.load_state_dict(common_state_dict)

        # global_user_embedding = global_user_embedding/user_index
        # user_embedding = torch.cat(cluster_num*[global_user_embedding])
        # global_user_embedding.unsqueeze(1).expand(latent_size, cluster_num)

        # network.m_user_embedding.data.copy_(user_embedding.data)

    def f_train(self, train_data, eval_data, pretrain_newtork, pretrain_optimizer, network, en_optimizer, de_optimizer, logger_obj):
        
        print("=="*10, "pretraining user item encoder", "=="*10)
        last_train_loss = 0
        last_val_loss = 0

        overfit_indicator = 0

        for epoch in range(self.m_epochs):
            print("++"*20, epoch, "++"*20)
            self.f_pretrain_epoch(train_data, pretrain_newtork, pretrain_optimizer, logger_obj)

            if last_train_loss == 0:
                last_train_loss = self.m_mean_pre_train_loss

            elif last_train_loss < self.m_mean_pre_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_en_train_loss)
                break
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_en_train_loss)
                last_train_loss = self.m_mean_pre_train_loss

        self.f_initialize_network(train_data, pretrain_newtork, network)

        last_train_loss = 0
        last_val_loss = 0

        overfit_indicator = 0

        for epoch in range(self.m_epochs):
            print("++"*20, epoch, "++"*20)

            s_time = datetime.datetime.now()
            self.f_train_en_epoch(train_data, network, en_optimizer, logger_obj)
            e_time = datetime.datetime.now()

            print("epoch duration", e_time-s_time)

            if last_train_loss == 0:
                last_train_loss = self.m_mean_en_train_loss

            elif last_train_loss < self.m_mean_en_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_en_train_loss)
                break
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_en_train_loss)
                last_train_loss = self.m_mean_en_train_loss

            print("++"*20, epoch, "++"*20)

            s_time = datetime.datetime.now()
            self.f_eval_en_epoch(train_data, network, en_optimizer, logger_obj)
            e_time = datetime.datetime.now()

            print("validation epoch duration", e_time-s_time)

            if last_val_loss == 0:
                last_val_loss = self.m_mean_en_val_loss

            elif last_val_loss < self.m_mean_en_val_loss:
                print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_val_loss, "cur val loss %.4f"%self.m_mean_en_val_loss)
                overfit_indicator += 1

                if overfit_indicator > self.m_overfit_epoch_threshold:
                    break
            else:
                print("last val loss %.4f"%last_val_loss, "cur val loss %.4f"%self.m_mean_en_val_loss)
                last_val_loss = self.m_mean_en_val_loss
        
        self.f_get_user_item(train_data, network, logger_obj)

        print("----"*10)
        last_train_loss = 0
        last_val_loss = 0
        overfit_indicator = 0
        for epoch in range(self.m_epochs):
            print("++"*20, epoch, "++"*20)

            s_time = datetime.datetime.now()
            self.f_train_de_epoch(train_data, network, de_optimizer, logger_obj)
            e_time = datetime.datetime.now()

            print("epoch duration", e_time-s_time)

            if last_train_loss == 0:
                last_train_loss = self.m_mean_de_train_loss

            elif last_train_loss < self.m_mean_de_train_loss:
                print("!"*10, "error training loss increase", "!"*10, "last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_de_train_loss)
                break
            else:
                print("last train loss %.4f"%last_train_loss, "cur train loss %.4f"%self.m_mean_de_train_loss)
                last_train_loss = self.m_mean_de_train_loss

            print("++"*20, epoch, "++"*20)

            s_time = datetime.datetime.now()
            self.f_eval_de_epoch(train_data, network, de_optimizer, logger_obj)
            e_time = datetime.datetime.now()

            print("validation epoch duration", e_time-s_time)

            if last_val_loss == 0:
                last_val_loss = self.m_mean_de_val_loss

            elif last_val_loss < self.m_mean_de_val_loss:
                print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_val_loss, "cur val loss %.4f"%self.m_mean_de_val_loss)

                overfit_indicator += 1

                if overfit_indicator > self.m_overfit_epoch_threshold:
                    break
            else:
                print("last val loss %.4f"%last_val_loss, "cur val loss %.4f"%self.m_mean_de_val_loss)
                last_val_loss = self.m_mean_de_val_loss

        self.f_save_model(epoch, network, en_optimizer, de_optimizer)

    def f_pretrain_epoch(self, train_data, network, optimizer, logger_obj):
        network.train()

        pre_train_loss_list = []
        en_NLL_loss_list = []
        iteration = 0

        for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in train_data:
            input_batch_gpu = input_batch.to(self.m_device)
            input_length_batch_gpu = input_length_batch.to(self.m_device)

            user_batch_gpu = user_batch.to(self.m_device)
            item_batch_gpu = item_batch.to(self.m_device)

            target_batch_gpu = target_batch.to(self.m_device)
            target_length_batch_gpu = target_length_batch.to(self.m_device)

            input_de_batch_gpu = target_batch[:, :-1]
            input_de_length_batch = target_length_batch-1

            batch_size = input_batch.size(0)
            # print("+++"*20)
            user_logits, item_logits = network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
            # user_logits, item_logits = network.module.m_user_item_encoder(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
            user_en_NLL_loss = self.m_rec_loss(user_logits, target_batch_gpu[:, 1:], target_length_batch-1)
            en_NLL_loss = user_en_NLL_loss/batch_size

            item_en_NLL_loss = self.m_rec_loss(item_logits, target_batch_gpu[:, 1:], target_length_batch-1)
            en_NLL_loss += item_en_NLL_loss/batch_size

            en_NLL_loss_list.append(en_NLL_loss.item())
            pre_train_loss_list.append(en_NLL_loss.item())
            
            optimizer.zero_grad()
            en_NLL_loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % self.m_print_interval == 0:
                
                logger_obj.f_add_output2IO("%d, pretrain en_NLL_loss:%.4f"%(iteration, np.mean(en_NLL_loss_list)))

                en_NLL_loss_list = []

        self.m_mean_pre_train_loss = np.mean(pre_train_loss_list)


    def f_train_en_epoch(self, train_data, network, en_optimizer, logger_obj):
        en_NLL_loss_list = []
        en_train_loss_list = []
        iteration = 0

        # logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)
        # logger_obj.f_add_output2IO("--"*20)

        network.train()
        for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in train_data:
            input_batch_gpu = input_batch.to(self.m_device)
            input_length_batch_gpu = input_length_batch.to(self.m_device)

            user_batch_gpu = user_batch.to(self.m_device)
            item_batch_gpu = item_batch.to(self.m_device)

            target_batch_gpu = target_batch.to(self.m_device)
            target_length_batch_gpu = target_length_batch.to(self.m_device)

            input_de_batch_gpu = target_batch[:, :-1]
            input_de_length_batch = target_length_batch-1

            batch_size = input_batch.size(0)
            # print("+++"*20)
            user_logits, item_logits, user_quantize_diff = network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag, "encode")
            # user_logits, item_logits = network.module.m_user_item_encoder(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
            user_en_NLL_loss = self.m_rec_loss(user_logits, target_batch_gpu[:, 1:], target_length_batch-1)
            en_NLL_loss = user_en_NLL_loss/batch_size

            en_NLL_loss += user_quantize_diff

            item_en_NLL_loss = self.m_rec_loss(item_logits, target_batch_gpu[:, 1:], target_length_batch-1)
            en_NLL_loss += item_en_NLL_loss/batch_size

            en_NLL_loss_list.append(en_NLL_loss.item())
            
            en_optimizer.zero_grad()
            en_NLL_loss.backward()
            en_optimizer.step()

            en_train_loss_list.append(en_NLL_loss.item())

            iteration += 1
            if iteration % self.m_print_interval == 0:
                
                logger_obj.f_add_output2IO("%d, en_NLL_loss:%.4f"%(iteration, np.mean(en_NLL_loss_list)))

                en_NLL_loss_list = []

        self.m_mean_en_train_loss = np.mean(en_train_loss_list)

    def f_eval_en_epoch(self, eval_data, network, en_optimizer, logger_obj):
        en_NLL_loss_list = []
        en_eval_loss_list = []

        iteration = 0

        # logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)
        # logger_obj.f_add_output2IO("--"*20)

        network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:

                eval_flag = random.randint(1,101)
                if eval_flag != 10:
                    continue

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch[:, :-1]
                input_de_length_batch = target_length_batch-1

                batch_size = input_batch.size(0)

                user_logits, item_logits, user_quantize_diff = network.m_user_item_encoder(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)
                user_en_NLL_loss = self.m_rec_loss(user_logits, target_batch_gpu[:, 1:], target_length_batch-1)
                en_NLL_loss = user_en_NLL_loss/batch_size

                item_en_NLL_loss = self.m_rec_loss(item_logits, target_batch_gpu[:, 1:], target_length_batch-1)
                en_NLL_loss += item_en_NLL_loss/batch_size

                en_NLL_loss_list.append(en_NLL_loss.item())

                iteration += 1
                if iteration % self.m_print_interval == 0:
                    
                    logger_obj.f_add_output2IO("%d, en_NLL_loss:%.4f"%(iteration, np.mean(en_NLL_loss_list)))

                    en_NLL_loss_list = []

                en_eval_loss_list.append(en_NLL_loss.item())

        self.m_mean_en_val_loss = np.mean(en_eval_loss_list)
        network.train()

    def f_get_user_item(self, data, network, logger_obj):
        logger_obj.f_add_output2IO(" "*10+"obtaining the user and item embedding"+" "*10)
        # logger_obj.f_add_output2IO("--"*20)

        s_time = datetime.datetime.now()
        
        network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in data:
                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                input_de_batch_gpu = target_batch[:, :-1]
                input_de_length_batch = target_length_batch-1
                
                network.update_user_item(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)

            network.normalize_user_item()

        e_time = datetime.datetime.now()
        print("get user & item representation duration", e_time-s_time)
        network.train()

    def f_train_de_epoch(self, train_data, network, de_optimizer, logger_obj):
        # logger_obj.f_add_output2IO("--"*20)
    
        gen_NLL_loss_list = []
        de_train_loss_list = []

        iteration = 0

        logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+"training the generator network"+" "*10)

        network.train()
        for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in train_data:
            input_batch_gpu = input_batch.to(self.m_device)
            input_length_batch_gpu = input_length_batch.to(self.m_device)

            user_batch_gpu = user_batch.to(self.m_device)
            item_batch_gpu = item_batch.to(self.m_device)

            target_batch_gpu = target_batch.to(self.m_device)
            target_length_batch_gpu = target_length_batch.to(self.m_device)

            # input_de_batch_gpu = target_batch[:, :-1]
            # input_de_length_batch = target_length_batch-1

            batch_size = input_batch.size(0)

            # logits = network.module.decode(input_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag)
            logits = network(input_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag, "decode")

            gen_NLL_loss = self.m_rec_loss(logits, target_batch_gpu[:, 1:], target_length_batch-1)

            gen_NLL_loss = gen_NLL_loss/batch_size
            gen_NLL_loss_list.append(gen_NLL_loss.item())

            de_optimizer.zero_grad()
            gen_NLL_loss.backward()
            de_optimizer.step()

            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO("%d, en_NLL_loss:%.4f"%(iteration, np.mean(gen_NLL_loss_list)))

            de_train_loss_list.append(gen_NLL_loss.item())

        self.m_mean_de_train_loss = np.mean(de_train_loss_list)
    
    def f_eval_de_epoch(self, eval_data, network, de_optimizer, logger_obj):
        
        gen_NLL_loss_list = []
        de_eval_loss_list = []
        iteration = 0

        logger_obj.f_add_output2IO("--"*20)
        logger_obj.f_add_output2IO(" "*10+"eval the generator network"+" "*10)

        network.eval()
        with torch.no_grad():
            for input_batch, input_length_batch, user_batch, item_batch, target_batch, target_length_batch, random_flag in eval_data:
                eval_flag = random.randint(1,101)
                if eval_flag != 10:
                    continue

                input_batch_gpu = input_batch.to(self.m_device)
                input_length_batch_gpu = input_length_batch.to(self.m_device)

                user_batch_gpu = user_batch.to(self.m_device)
                item_batch_gpu = item_batch.to(self.m_device)

                target_batch_gpu = target_batch.to(self.m_device)
                target_length_batch_gpu = target_length_batch.to(self.m_device)

                # input_de_batch_gpu = target_batch[:, :-1]
                # input_de_length_batch = target_length_batch-1

                batch_size = input_batch.size(0)

                logits = network.decode(input_batch_gpu, user_batch_gpu, item_batch_gpu, random_flag)

                gen_NLL_loss = self.m_rec_loss(logits, target_batch_gpu[:, 1:], target_length_batch-1)

                gen_NLL_loss = gen_NLL_loss/batch_size
                gen_NLL_loss_list.append(gen_NLL_loss.item())

                iteration += 1
                if iteration % self.m_print_interval == 0:
                    logger_obj.f_add_output2IO("%d, en_NLL_loss:%.4f"%(iteration, np.mean(gen_NLL_loss_list)))

                de_eval_loss_list.append(gen_NLL_loss.item())

            self.m_mean_de_val_loss = np.mean(de_eval_loss_list)
        network.train()
