import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Reconstruction_loss(nn.Module):
    def __init__(self, device, ignore_index):
        super(Reconstruction_loss, self).__init__()
        self.m_device = device
        self.m_NLL = nn.NLLLoss(size_average=False, ignore_index=ignore_index).to(self.m_device)

    def forward(self, pred, target, length):
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        pred = pred.view(-1, pred.size(2))

        NLL_loss = self.m_NLL(pred, target)
        return NLL_loss

class KL_loss_z(nn.Module):
    def __init__(self, device):
        super(KL_loss_z, self).__init__()
        print("kl loss for z")

        self.m_device = device
    
    # def forward(self, mean_prior, logv_prior, mean, logv, step, k, x0, anneal_func):
    def forward(self, mean_prior, mean, logv, step, k, x0, anneal_func):
        # loss = -0.5*torch.sum(-logv_prior+logv+1-logv.exp()/logv_prior.exp()-((mean_prior-mean)/logv_prior.exp()).pow(2))

        loss = -0.5*torch.sum(logv+1-logv.exp()-(mean_prior-mean).pow(2))

        weight = 0
        if anneal_func == "logistic":
            weight = float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_func == "":
            weight = min(1, step/x0)
        else:
            raise NotImplementedError

        return loss, weight

class KL_loss(nn.Module):
    def __init__(self, device):
        super(KL_loss, self).__init__()
        print("kl loss")

        self.m_device = device

    def forward(self, mean, logv, step, k, x0, anneal_func):
        loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        weight = 0
        if anneal_func == "logistic":
            weight = float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_func == "":
            weight = min(1, step/x0)
        else:
            raise NotImplementedError

        return loss, weight

class RRe_loss(nn.Module):
    def __init__(self, device):
        super(RRe_loss, self).__init__()
        self.m_device = device
        # self.m_loss_fun = 
        # self.m_BCE = nn.BCELoss().to(self.m_device)
        self.m_logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, target):
        
        loss = torch.mean(torch.sum(-target*self.m_logsoftmax(pred), dim=1))
        # RRe_loss = self.m_BCE(pred, target)
        # exit()
        return loss

class ARe_loss(nn.Module):
    def __init__(self, device):
        super(ARe_loss, self).__init__()
        self.m_device = device
        self.m_logsoftmax = nn.LogSoftmax(dim=1)
        # self.m_BCE = nn.BCELoss().to(self.m_device)
    
    def forward(self, pred, target):
        # print("pred", pred.size())
        # print("target", target.size())
        loss = torch.mean(torch.sum(-target*self.m_logsoftmax(pred), dim=1))
        # loss = self.m_BCE(pred, target)

        return loss


# class BLEU()