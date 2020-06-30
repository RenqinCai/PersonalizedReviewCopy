import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 

class _REC_LOSS(nn.Module):
    def __init__(self, ignore_index, device, reduction=False):
        super(_REC_LOSS, self).__init__()
        self.m_device = device
        if reduction == True:
            self.m_NLL = nn.NLLLoss(ignore_index=ignore_index, reduction="none").to(self.m_device)
        else:
            self.m_NLL = nn.NLLLoss(size_average=False, ignore_index=ignore_index).to(self.m_device)

    def forward(self, pred, target, length):
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        target = target.contiguous().view(-1)

        pred = pred.view(-1, pred.size(2))
        # print("target", target)
        # print("pred", pred)

        NLL_loss = self.m_NLL(pred, target)
        return NLL_loss

class _KL_LOSS_CUSTOMIZE(nn.Module):
    def __init__(self, device):
        super(_KL_LOSS_CUSTOMIZE, self).__init__()
        print("kl loss with non zero prior")

        self.m_device = device
    
    # def forward(self, mean_prior, logv_prior, mean, logv, step, k, x0, anneal_func):
    def forward(self, mean_prior, mean, logv):
        # loss = -0.5*torch.sum(-logv_prior+logv+1-logv.exp()/logv_prior.exp()-((mean_prior-mean)/logv_prior.exp()).pow(2))

        loss = -0.5*torch.sum(logv+1-logv.exp()-(mean-mean_prior).pow(2))

        return loss 

class _KL_LOSS_STANDARD(nn.Module):
    def __init__(self, device, reduction=False):
        super(_KL_LOSS_STANDARD, self).__init__()
        print("kl loss")

        self.m_reduction = reduction
        
        self.m_device = device

    def forward(self, mean, logv):
        loss = 0
        if self.m_reduction:
            loss = -0.5 * torch.sum((1 + logv - mean.pow(2) - logv.exp()), dim=1)
        else:
            loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        return loss

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

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))

    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])

        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])

        stats.append(max([sum((s_ngrams&r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))

    return stats

def bleu(stats):
    if len(list(filter(lambda x:x==0, stats))) > 0:
        return 0

    (c, r) = stats[:2]
    log_bleu_prec = sum([np.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])])/4

    return np.exp(min([0, 1-float(r)/c])+log_bleu_prec)

def get_bleu(hypotheses, reference):
    stats = np.zeros(10)

    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    
    return 100*bleu(stats)


# class BLEU

# class BLEU()