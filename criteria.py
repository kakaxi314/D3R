#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import torch
import torch.nn as nn

__all__ = [
    'PSNR',
    'MSE',
]


def MSE():
    return nn.MSELoss()


class PSNR(nn.Module):

    def __init__(self):
        super().__init__()
        self.R = 255.

    def forward(self, outputs, target):
        outputs[outputs < 0] = 0
        outputs[outputs > 1] = 1
        outputs = (outputs * self.R).round().float()
        target = target * self.R
        mse = (outputs - target) ** 2
        mse = torch.mean(mse, dim=[1, 2, 3])
        acc = 10 * torch.log10(self.R ** 2 / mse)
        # For the INF case
        acc[acc > 100] = 100
        return acc.mean()
