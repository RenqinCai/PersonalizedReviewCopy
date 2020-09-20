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
        
        if torch.isnan(preds).any():
            print("preds", preds)

        if torch.isnan(targets).any():
            print("targets", targets)

        len_mask = ~mask
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

        loss = F.logsigmoid(logits*logit_mask)
    
        valid_mask = logit_mask*pos_mask
        valid_mask = valid_mask**2
        
        loss = loss*valid_mask

        loss = torch.sum(loss)
        logit_num = torch.sum(valid_mask)

        loss = -loss/logit_num

        # print("loss", loss.item())

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

class _ARE_LOSS(nn.Module):
    def __init__(self, device):
        super(_ARE_LOSS, self).__init__()
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

def get_recall(preds, targets, k=10):
    
    batch_size = preds.shape[0]
    voc_size = preds.shape[1]

    print("batch_size", batch_size)
    print("voc_size", voc_size)

    idx = bn.argpartition(-preds, k, axis=1)
    hit = targets[np.arange(batch_size)[:, np.newaxis], idx[:, :k]]

    hit = np.count_nonzero(hit, axis=1)
    hit = np.array(hit)
    
    hit = np.squeeze(hit)

    recall = np.array([min(n, k) for n in np.count_nonzero(targets, axis=1)])

    recall = hit/recall

    return recall

def get_precision_recall(preds, targets, k=1):
    preds = preds.view(-1, preds.size(1))
    _, indices = torch.topk(preds, k, -1)

    pos = torch.sum(targets, dim=1)

    if pos.nonzero().size(0) != len(pos):
        # print("error")
        # print(pos)
        return 0, 0
    # recall_list = []
    # precision_list = []

    # for i, pos_i in enumerate(pos):
    #     nonzero_num_i = pos[i]
    #     indicies_i = indices[i][:nonzero_num_i]
    #     targets_i = targets[i]

    #     true_pos_i = targets_i[indicies_i]
    #     true_pos_i = torch.sum(true_pos_i)
    #     true_pos_i = true_pos_i.float()

    #     recall_i = true_pos_i/pos_i
    #     recall_list.append(recall_i)

    #     precision_i = true_pos_i/nonzero_num_i
    #     precision_list.append(precision_i)

    # recall = np.mean(recall_list)
    # precision = np.mean(precision_list)

    true_pos = torch.gather(targets, 1, indices)
    true_pos = torch.sum(true_pos, dim=1)
    true_pos = true_pos.float()

    recall = true_pos/pos
    precision = true_pos/k

    recall = torch.mean(recall)
    precision = torch.mean(precision)

    return precision, recall