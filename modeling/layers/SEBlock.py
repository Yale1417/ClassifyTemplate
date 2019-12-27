# -*- coding: utf-8 -*-
# @Time    : 2019-12-05 07:50
# @Author  : yangqiang
# @FileName: SEBlock.py
# @Blog    ：http://zhifei.online
import torch
import torch.nn as nn
from .activation import Swish


class SqueezeExcitation(nn.Module):
    """
     inplanes / se_planes = r>1
     先压缩s，再扩张e
     作者尝试了r在各种取值下的性能 ，最后得出结论r=16时整体性能和计算量最平衡。
    """
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            # nn.ReLU(),
            nn.Conv2d(se_planes, inplanes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x
