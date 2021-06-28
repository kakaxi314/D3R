#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    augs.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import numpy as np

__all__ = [
    'Compose',
    'HFlip',
    'VFlip',
    'Rotate',
    'Ident',
]


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb):
        for t in self.transforms:
            rgb = t(rgb)
        return rgb


class HFlip(object):
    """
    flip along horizontal axis
    """

    def __call__(self, rgb):
        flip = bool(np.random.randint(2))
        if flip:
            rgb = rgb[:, ::-1, :]
        return rgb


class VFlip(object):
    """
    flip along vertical axis
    """

    def __call__(self, rgb):
        flip = bool(np.random.randint(2))
        if flip:
            rgb = rgb[::-1, :, :]
        return rgb


class Rotate(object):
    """
    rotate 90 degrees
    """

    def __call__(self, rgb):
        rotate = bool(np.random.randint(2))
        if rotate:
            rgb = np.rot90(rgb)
        return rgb


class Ident(object):

    def __call__(self, rgb):
        return rgb
