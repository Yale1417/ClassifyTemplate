# -*- coding: utf-8 -*-
# @Time    : 2019-12-05 07:52
# @Author  : yangqiang
# @FileName: activate.py
# @Blog    ：http://zhifei.online
import torch.nn as nn
import torch
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))
