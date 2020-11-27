import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn
from torch.nn.modules import sparse

class PAIRWISE_LOSS(nn.Module):
    def __init__(self, device):
        super(PAIRWISE_LOSS, self).__init__()

        self.m_device = device

    def f_pairwise_and(self, a, b):
        # print("a", a)
        # print("b", b)
        column = a.unsqueeze(-1)
        row = b.unsqueeze(1)

        c = torch.logical_and(column, row)

        return c

    def f_pairwise_sub(self, a, b):
        column = a.unsqueeze(-1)
        row = b.unsqueeze(1)

        c = column - row

        return c

    def forward(self, preds, targets):
        # pred_size = preds.size()
        target_size = targets.size()

        pos_label = torch.eq(targets, torch.ones(target_size).to(self.m_device))
        neg_label = torch.eq(targets, torch.zeros(target_size).to(self.m_device))

        truth_matrix = self.f_pairwise_and(pos_label, neg_label).float()

        sub_matrix = self.f_pairwise_sub(preds, preds)

        exp_matrix = torch.exp(-5*sub_matrix)

        sparse_matrix = exp_matrix*truth_matrix

        sums = torch.sum(sparse_matrix, dim=[1, 2])

        pos_label_num = torch.sum(pos_label, dim=1)
        neg_label_num = torch.sum(neg_label, dim=1)

        normalizers = pos_label_num*neg_label_num

        loss = sums/(5.0*normalizers)

        zero_pad = torch.zeros_like(loss)
        flag = torch.logical_or(torch.isinf(loss), torch.isnan(loss))
        # print("flag", flag)
        # print("loss", loss)
        loss = torch.where(flag, zero_pad, loss)

        loss = torch.sum(loss)

        return loss

class BCE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(BCE_LOSS, self).__init__()

        self.m_bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.m_voc_size = voc_size
        self.m_device = device
    
    def forward(self, preds, targets):
        # print("=="*10)

        # print(targets)
        # targets = targets.long()
        # targets = F.one_hot(targets, self.m_voc_size)

        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0
        targets = targets.float()

        # print(targets.size())
        # print(preds.size())

        loss = self.m_bce_loss(preds, targets)

        batch_size = preds.size(0)
        loss = loss/batch_size

        return loss

class XE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(XE_LOSS, self).__init__()
        self.m_voc_size = voc_size
        self.m_device = device
    
    def forward(self, preds, targets):
        # print("==="*10)
        # print(targets.size())
        targets = F.one_hot(targets, self.m_voc_size)

        # print(targets.size())
        targets = torch.sum(targets, dim=1)

        # print(targets.size())

        targets[:, 0] = 0

        preds = F.log_softmax(preds, 1)
        xe_loss = torch.sum(preds*targets, dim=-1)

        xe_loss = -torch.mean(xe_loss)

        return xe_loss

class _REC_LOSS(nn.Module):
    def __init__(self, ignore_index, device):
        super(_REC_LOSS, self).__init__()
        self.m_device = device
        self.m_NLL = nn.NLLLoss(ignore_index=ignore_index).to(self.m_device)

    def forward(self, preds, targets, mask):
        pred_probs = F.log_softmax(preds.view(-1, preds.size(1)), dim=-1)
        targets = targets.contiguous().view(-1)

        NLL_loss = self.m_NLL(pred_probs, targets)
        NLL_loss = torch.mean(NLL_loss)
        return NLL_loss

class _REC_BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(_REC_BPR_LOSS, self).__init__()
        self.m_device = device
    
    def forward(self, preds, targets, mask):
        preds = preds.view(-1, preds.size(1))
        targets = targets.float()

        len_mask = mask
        len_mask = len_mask.int()
    
        batch_len = preds.size(1)

        logits = []
        logit_mask = []

        pos_mask = []

        for i in range(batch_len-1):
            logit_delta = preds[:, i].unsqueeze(1)-preds[:, i+1:]
            logits.append(logit_delta)

            mask_delta = targets[:, i].unsqueeze(1)-targets[:, i+1:]
            logit_mask.append(mask_delta)

            len_delta = len_mask[:, i].unsqueeze(1) & len_mask[:, i+1:]
            pos_mask.append(len_delta)

        logits = torch.cat(logits, dim=1)

        logit_mask = torch.cat(logit_mask, dim=1)

        pos_mask = torch.cat(pos_mask, dim=1)

        # print("logits ...", logits)
        # print("logit_mask ...", logit_mask)
        # print("pos_mask ...", pos_mask)

        loss = F.logsigmoid(logits*logit_mask)

        valid_mask = logit_mask*pos_mask
        valid_mask = valid_mask**2
        
        loss = loss*valid_mask

        loss = torch.sum(loss)
        logit_num = torch.sum(valid_mask)

        # print("valid_mask", valid_mask)

        # print("xxx loss", loss.item())

        # print("logit_num", logit_num.item())

        loss = -loss
        # loss = -loss/logit_num

        # print("loss", loss.item())

        # exit()

        return loss

# class NORM_LOSS(nn.Module):
#     def __init__(self, device):
#         super(NORM_LOSS, self).__init__()
#         self.m_device = device

#     def f_square_sum(self, input):
#         # return torch.sum(torch.square(input))
#         return torch.sum(input**2)

#     def forward(self, network):

#         """x"""
#         reg_x = self.f_square_sum(network.m_user_embedding.weight)
#         reg_x += self.f_square_sum(network.m_item_embedding.weight)
#         reg_x += self.f_square_sum(network.m_Fx_linear_1.weight)
#         reg_x += self.f_square_sum(network.m_Fx_linear_2.weight)

#         """enc"""
#         reg_enc = self.f_square_sum(network.m_attr.weight)
#         reg_enc += self.f_square_sum(network.m_Fe_linear_1.weight)

#         """dec"""
#         reg_dec = self.f_square_sum(network.m_Fd_linear_0.weight)
#         reg_dec += self.f_square_sum(network.m_Fd_linear_1.weight)

#         reg = reg_x+reg_enc+reg_dec

#         return reg

class NORM_LOSS(nn.Module):
    def __init__(self, device):
        super(NORM_LOSS, self).__init__()
        self.m_device = device

    def f_square_sum(self, input):
        # return torch.sum(torch.square(input))
        return torch.sum(input**2)

    def forward(self, network):

        """x"""

        reg_x = self.f_square_sum(network.m_user_embed.weight) 
        reg_x += self.f_square_sum(network.m_item_embed.weight)

        # reg_x = self.f_square_sum(network.m_Fx_w0)
        # # reg_x += self.f_square_sum(network.m_Fx_b0)
        # reg_x += self.f_square_sum(network.m_Fx_w1)
        # # reg_x += self.f_square_sum(network.m_Fx_b1)
        reg_x += self.f_square_sum(network.m_Fx_w2)
        # reg_x += self.f_square_sum(network.m_Fx_b2)

        """enc"""
        reg_enc = self.f_square_sum(network.m_Fe_w0)
        # reg_enc += self.f_square_sum(network.m_Fe_b0)
        reg_enc += self.f_square_sum(network.m_Fe_w1)
        # reg_enc += self.f_square_sum(network.m_Fe_b1)

        """dec"""
        reg_dec = self.f_square_sum(network.m_Fd_w0)
        # reg_dec += self.f_square_sum(network.m_Fd_b0)

        reg_dec += self.f_square_sum(network.m_Fd_w1)
        # reg_dec += self.f_square_sum(network.m_Fd_b1)

        reg = reg_x+reg_enc+reg_dec

        return reg

class EMBED_LOSS(nn.Module):
    def __init__(self, device):
        super(EMBED_LOSS, self).__init__()
        self.m_device = device

        self.m_lagrange = 0.5

    def forward(self, enc_x, enc_y):
        I = torch.eye(enc_x.size()[1]).to(enc_x.device)
        C1 = enc_x-enc_y
    
        C2 = torch.matmul(enc_x.t(), enc_x)-I

        C3 = torch.matmul(enc_y.t(), enc_y)-I

        # print("C1", C1)
        # print("C2", C2)
        # print("C3", C3)

        loss = torch.trace(torch.matmul(C1, C1.t()))+self.m_lagrange*torch.trace(torch.matmul(C2, C2.t())+torch.matmul(C3, C3.t()))

        return loss
