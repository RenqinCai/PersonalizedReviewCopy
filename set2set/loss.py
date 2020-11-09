import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

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

class _REC_SOFTMAX_BOW_LOSS(nn.Module):
    def __init__(self, device):
        super(_REC_SOFTMAX_BOW_LOSS, self).__init__()
        self.m_device = device
        
    def forward(self, preds, targets, mask):
        # preds: batch_size*seq_len, logits
        # targets: batch_size*seq_len, {0, 1}

        preds = preds.view(-1, preds.size(1))
        mask = ~mask
        if torch.isnan(preds).any():
            print("preds", preds)

        # preds[~mask] = float('-inf')
        preds = F.softmax(preds, dim=1)

        # preds = preds+1e-20
        log_preds = torch.log(preds)

        rec_loss = torch.sum(log_preds*targets*mask, dim=-1)
        rec_loss = -torch.mean(rec_loss)

        return rec_loss

class _REC_BOW_LOSS(nn.Module):
    def __init__(self, device):
        super(_REC_BOW_LOSS, self).__init__()
        self.m_device = device
        self.m_bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets, mask):
        ## preds: batch_size*seq_len
        preds = preds.view(-1, preds.size(1))
        targets = targets.float()
        # preds = F.log_softmax(preds.view(-1, preds.size(1)), dim=1)
        mask = ~mask
        rec_loss = self.m_bce_loss(preds, targets)
        rec_loss = torch.sum(rec_loss*mask, dim=-1)
        # rec_loss = torch.sum(preds*targets*mask, dim=-1)

        rec_loss = torch.mean(rec_loss)

        return rec_loss

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

        # if torch.isnan(loss).any():
        #     print("loss nan", loss)
        
        # print(debug.size())
        # loss_debug = loss*debug
        
        # print("loss debug")
        # print(loss_debug)
        # print("loss", loss)

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

class _REC_BPR_LOSS_IPS(nn.Module):
    def __init__(self, device):
        super(_REC_BPR_LOSS_IPS, self).__init__()
        self.m_device = device
    
    def forward(self, preds, targets, mask, attr_tf):
        preds = preds.view(-1, preds.size(1))
        targets = targets.float()
        
        len_mask = ~mask
        len_mask = len_mask.int()
    
        batch_len = preds.size(1)

        logits = []
        logit_mask = []

        pos_mask = []

        logit_ips = []

        for i in range(batch_len-1):
            logit_delta = preds[:, i].unsqueeze(1)-preds[:, i+1:]
            logits.append(logit_delta)

            mask_delta = targets[:, i].unsqueeze(1)-targets[:, i+1:]
            logit_mask.append(mask_delta)

            len_delta = len_mask[:, i].unsqueeze(1) & len_mask[:, i+1:]
            pos_mask.append(len_delta)

            ips_delta = attr_tf[:, i].unsqueeze(1)-attr_tf[:, i+1:]
            logit_ips.append(ips_delta)
            
        logits = torch.cat(logits, dim=1)

        logit_mask = torch.cat(logit_mask, dim=1)

        pos_mask = torch.cat(pos_mask, dim=1)

        logit_ips = torch.cat(logit_ips, dim=1)

        loss = F.logsigmoid(logits*logit_mask-torch.log(logit_ips))
    
        valid_mask = logit_mask*pos_mask
        valid_mask = valid_mask**2
        
        loss = loss*valid_mask

        loss = torch.sum(loss)
        logit_num = torch.sum(valid_mask)

        loss = -loss/logit_num

        return loss

class _KL_LOSS_CUSTOMIZE(nn.Module):
    def __init__(self, device):
        super(_KL_LOSS_CUSTOMIZE, self).__init__()
        print("kl loss with non zero prior")

        self.m_device = device
    
    def forward(self, mean_prior, logv_prior, mean, logv):
    # def forward(self, mean_prior, mean, logv):
        loss = -0.5*torch.sum(-logv_prior+logv+1-logv.exp()/logv_prior.exp()-((mean_prior-mean).pow(2)/logv_prior.exp()))

        # loss = -0.5*torch.sum(logv+1-logv.exp()-(mean-mean_prior).pow(2))

        return loss 

class _KL_LOSS_STANDARD(nn.Module):
    def __init__(self, device):
        super(_KL_LOSS_STANDARD, self).__init__()
        print("kl loss")

        self.m_device = device

    def forward(self, mean, logv):
        loss = -0.5*torch.sum(1+logv-mean.pow(2)-logv.exp())

        return loss

class _RRE_LOSS(nn.Module):
    def __init__(self, device):
        super(_RRE_LOSS, self).__init__()
        self.m_device = device
        # self.m_loss_fun = 
        # self.m_BCE = nn.BCELoss().to(self.m_device)
        self.m_logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, target):
        
        loss = torch.mean(torch.sum(-target*self.m_logsoftmax(pred), dim=1))
        # RRe_loss = self.m_BCE(pred, target)
        # exit()
        return loss

class _BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(_BPR_LOSS, self).__init__()
        self.m_device = device

    def forward(self, logits):
        ### logits: batch_size

        # loss = torch.sigmoid(logits)
        # # print("loss 1", loss)
        # loss = torch.log(loss)
        loss = F.logsigmoid(logits)
        # print("loss 2", loss)
        loss = -torch.sum(loss)

        return loss