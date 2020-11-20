import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

class SEQ_XE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(SEQ_XE_LOSS, self).__init__()
        self.m_voc_size = voc_size

        self.m_xe_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.m_device = device

    def f_generate_mask(self, length):
        max_len = length.max().item()
        # print("max_len", max_len)
        mask = torch.arange(0, max_len).expand(len(length), max_len).to(length.device)
        mask = mask < length.unsqueeze(1)

        return mask

    def forward(self, preds, targets, target_lens):

        target_mask = self.f_generate_mask(target_lens)

        mask_preds = preds[target_mask]
        mask_targets = targets[target_mask]
        
        mask_preds = mask_preds.view(-1, mask_preds.size(-1))
        loss = self.m_xe_loss(mask_preds, mask_targets.contiguous().view(-1))

        loss = torch.mean(loss)

        return loss

class XE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(XE_LOSS, self).__init__()
        self.m_voc_size = voc_size
        self.m_device = device
    
    def forward(self, preds, targets):

        # print("targets", targets.size())

        targets = F.one_hot(targets, self.m_voc_size)
        # print("targets", targets.size())

        targets = torch.sum(targets, dim=1)

        # print(targets.size())

        targets[:, 0] = 0

        preds = F.log_softmax(preds, 1)

        # print("preds size", preds.size())
        xe_loss = torch.sum(preds*targets, dim=-1)

        xe_loss = -torch.mean(xe_loss)

        return xe_loss

class MASK_XE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(MASK_XE_LOSS, self).__init__()
        self.m_voc_size = voc_size
        self.m_device = device
    
    def forward(self, preds, targets, pos_num):

        targets = F.one_hot(targets, self.m_voc_size)
        # print("targets", targets.size())

        # targets = torch.sum(targets, dim=1)

        # print(targets.size())

        targets[:, :, 0] = 0

        preds = F.log_softmax(preds, 1)

        preds = preds.unsqueeze(1)

        # print("preds", preds.size())

        xe_loss = 0
        loss_num = 0
        for i, pos_num_i in enumerate(pos_num):
                
            # print(preds[i].size())
            # print(targets[i, :pos_num_i].size())
            xe_loss_i = torch.sum(preds[i]*targets[i, :pos_num_i], dim=-1)

            loss_num += xe_loss_i.size(0)

            xe_loss_i = torch.sum(xe_loss_i)
            
            xe_loss += xe_loss_i

        xe_loss = -xe_loss/loss_num
        # xe_loss = -torch.mean(xe_loss)
        # print("xe_loss", xe_loss)
        return xe_loss

class BCE_LOSS(nn.Module):
    def __init__(self, voc_size, device):
        super(BCE_LOSS, self).__init__()

        self.m_bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.m_voc_size = voc_size
        self.m_device = device
    
    def forward(self, preds, targets):
        # print("=="*10)
        targets = F.one_hot(targets, self.m_voc_size)

        targets = torch.sum(targets, dim=1)

        targets[:, 0] = 0
        targets = targets.float()

        # print("preds", preds)
        # print("targets", targets)
        # preds = torch.sigmoid(preds)

        loss = self.m_bce_loss(preds, targets)

        batch_size = preds.size(0)
        loss = loss/batch_size

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

class BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(BPR_LOSS, self).__init__()
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

class BPR_LOSS_COND(nn.Module):
    def __init__(self, device):
        super(BPR_LOSS_COND, self).__init__()
        self.m_device = device

    def forward(self, cond_logits, logits):
        ### logits: batch_size

        # loss = torch.sigmoid(logits)
        # # print("loss 1", loss)
        # loss = torch.log(loss)
        loss = F.logsigmoid(logits)
        # print("loss 2", loss)
        loss = -torch.sum(loss)

        cond_loss = F.logsigmoid(cond_logits)
        # print("loss 2", loss)
        cond_loss = -torch.sum(cond_loss)

        loss = loss + cond_loss

        return loss