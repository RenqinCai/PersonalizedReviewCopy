import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter 
import bottleneck as bn

class _BPR_LOSS(nn.Module):
    def __init__(self, device):
        super(_BPR_LOSS, self).__init__()
        self.m_device = device

    def forward(self, logits):
        ### logits: batch_size

        loss = torch.sigmoid(logits)
        # print("loss 1", loss)
        loss = torch.log(loss)
        # print("loss 2", loss)
        loss = -torch.mean(loss)

        return loss