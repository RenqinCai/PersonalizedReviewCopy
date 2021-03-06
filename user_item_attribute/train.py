import os
import json
import time
from yelp_restaurant import pretrain_word2vec
import torch
import argparse
import numpy as np
import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from metric import _REC_LOSS, _REC_BOW_LOSS, _KL_LOSS_CUSTOMIZE, _KL_LOSS_STANDARD, _RRE_LOSS, _ARE_LOSS, _REC_SOFTMAX_BOW_LOSS, get_precision_recall, _REC_BPR_LOSS
from model import _ATTR_NETWORK
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
		
		self.m_epochs = args.epoch_num
		self.m_batch_size = args.batch_size

		# self.m_x0 = args.x0
		# self.m_k = args.k
		
		# self.m_rec_loss = _REC_BOW_LOSS(self.m_device)
		# self.m_rec_loss = _REC_SOFTMAX_BOW_LOSS(self.m_device)
		# self.m_rec_loss = _REC_LOSS(self.m_pad_idx, self.m_device)
		self.m_rec_loss = _REC_BPR_LOSS(self.m_device)

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
		# self.f_init_word_embed(pretrain_word_embed, network)
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

				# print("--"*10, epoch, "--"*10)

				# s_time = datetime.datetime.now()
				# self.f_eval_epoch(eval_data, network, optimizer, logger_obj)
				# e_time = datetime.datetime.now()

				# print("validation epoch duration", e_time-s_time)

				# if last_eval_loss == 0:
				# 	last_eval_loss = self.m_mean_eval_loss

				# elif last_eval_loss < self.m_mean_eval_loss:
				# 	print("!"*10, "error val loss increase", "!"*10, "last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
					
				# 	overfit_indicator += 1

				# 	# if overfit_indicator > self.m_overfit_epoch_threshold:
				# 	# 	break
				# else:
				# 	print("last val loss %.4f"%last_eval_loss, "cur val loss %.4f"%self.m_mean_eval_loss)
				# 	last_eval_loss = self.m_mean_eval_loss

				if best_eval_precision < self.m_mean_eval_precision:
					checkpoint = {'model':network.state_dict()}
					self.f_save_model(checkpoint)
					best_eval_precision = self.m_mean_eval_precision

		except KeyboardInterrupt:
			print("--"*20)
			print("... exiting from training early") 
			if best_eval_precision < self.m_mean_eval_precision:
				checkpoint = {'model':network.state_dict()}
				self.f_save_model(checkpoint)
				best_eval_precision = self.m_mean_eval_precision

		# self.f_get_user_item(train_data, network, logger_obj, rank)

		# if rank == 0:
			# checkpoint = {'model':network.module.state_dict(),
			# 	'epoch': epoch,
			# 	'optimizer': optimizer
			# }
			

	def f_train_epoch(self, train_data, network, optimizer, logger_obj):
		loss_list = []
		precision_list = []
		recall_list = []

		iteration = 0

		# logger_obj.f_add_output2IO("--"*20)
		logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)
		# logger_obj.f_add_output2IO("--"*20)

		beta = 0.1

		tmp_loss_list = []
		tmp_precision_list = []
		tmp_recall_list = []

		network.train()
		for input_batch, input_freq_batch, input_length_batch, user_batch, item_batch, target_batch in train_data:
			input_batch_gpu = input_batch.to(self.m_device)
			
			input_freq_batch_gpu = input_freq_batch.to(self.m_device)
			input_length_batch_gpu = input_length_batch.to(self.m_device)

			user_batch_gpu = user_batch.to(self.m_device)
			item_batch_gpu = item_batch.to(self.m_device)

			target_batch_gpu = target_batch.to(self.m_device)

			batch_size = input_batch.size(0)
			# print("+++"*20)
			# logits, z, z_mean, z_logvar = network(input_batch_gpu)
			user_item_attr_logits, mask = network(input_batch_gpu, input_freq_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)

			NLL_loss = self.m_rec_loss(user_item_attr_logits, target_batch_gpu, mask)
			loss = NLL_loss

			precision, recall = get_precision_recall(user_item_attr_logits.cpu(), target_batch, k=3)
			if precision != 0 and recall != 0:
				loss_list.append(loss.item()) 
				precision_list.append(precision)
				recall_list.append(recall)

				tmp_loss_list.append(loss.item())
				tmp_precision_list.append(precision)
				tmp_recall_list.append(recall)
				
			# loss = NLL_loss+beta*KL_loss
			
			# print("loss", loss)

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
				
			# 	logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)
			# 	logger_obj.f_add_scalar2tensorboard("train/precision", np.mean(precision_list), self.m_train_iteration)
			# 	logger_obj.f_add_scalar2tensorboard("train/recall", np.mean(recall_list), self.m_train_iteration)
			
			# 	loss_list = []
			# 	precision_list = []
			# 	recall_list = []
			
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

		iteration = 0
		# self.m_eval_iteration = 0
		self.m_eval_iteration = self.m_train_iteration

		# logger_obj.f_add_output2IO("--"*20)
		logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)
		# logger_obj.f_add_output2IO("--"*20)

		beta = 0.1
		network.eval()
		with torch.no_grad():
			for input_batch, input_freq_batch, input_length_batch, user_batch, item_batch, target_batch in eval_data:

				# eval_flag = random.randint(1,5)
				# if eval_flag != 2:
				# 	continue

				input_batch_gpu = input_batch.to(self.m_device)
				input_freq_batch_gpu = input_freq_batch.to(self.m_device)
				input_length_batch_gpu = input_length_batch.to(self.m_device)

				user_batch_gpu = user_batch.to(self.m_device)
				item_batch_gpu = item_batch.to(self.m_device)

				target_batch_gpu = target_batch.to(self.m_device)

				batch_size = input_batch.size(0)

				user_item_attr_logits, mask = network(input_batch_gpu, input_freq_batch_gpu, input_length_batch_gpu, user_batch_gpu, item_batch_gpu)

				# target_batch_gpu = torch.gather(target_batch_gpu, 1, input_batch_gpu)

				NLL_loss = self.m_rec_loss(user_item_attr_logits, target_batch_gpu, mask)
				
				# NLL_loss = NLL_loss/batch_size

				loss = NLL_loss

				precision, recall = get_precision_recall(user_item_attr_logits.cpu(), target_batch, k=3)
				# print("precision", precision, "recall", recall)

				if precision != 0 and recall != 0:
					loss_list.append(loss.item()) 
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

