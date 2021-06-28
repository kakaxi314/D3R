#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    datasets.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import torch.utils.data as data
import numpy as np
import os
import lmdb
import pyarrow as pa
import math

__all__ = [
    'color_lmdb',
]


class color_lmdb(data.Dataset):

    def __init__(self, path='datas/color', mode='train_lmdb', train=True, uf=8, pf=4, transform=None, noise=None,
                 return_idx=False):
        """ The size of input image should be multiple of both pattern size and U-Net factor
        :param pf: pattern factor
        :param uf: U-Net factor
        """
        self.lcm = uf * pf // math.gcd(uf, pf)
        self.transform = transform
        self.return_idx = return_idx
        self.noise = noise
        self.train = train
        lmdb_path = os.path.join(path, mode)
        #############################################################
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))
            self.names = pa.deserialize(txn.get(b'__names__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        rgb, = unpacked
        rgb = rgb.astype(np.float32) / 255.
        if self.transform:
            rgb = self.transform(rgb)
        gt = rgb.transpose(2, 0, 1).astype(np.float32)
        h, w, c = rgb.shape
        fh = math.ceil(h / float(self.lcm)) * self.lcm
        fw = math.ceil(w / float(self.lcm)) * self.lcm
        wl = (fw - w) // 2
        wr = fw - w - wl
        hl = (fh - h) // 2
        hr = fh - h - hl
        rgb_pad = np.pad(rgb, pad_width=((hl, hr), (wl, wr), (0, 0)), mode='edge')
        rgb_pad = rgb_pad.transpose(2, 0, 1).astype(np.float32)
        rgbn = rgb_pad.copy()
        if self.noise:
            if isinstance(self.noise, (list, tuple)):
                noise = self.noise[np.random.randint(len(self.noise))]
            else:
                noise = self.noise
            rgbn = rgbn + np.random.normal(0, noise / 255.0, rgbn.shape).astype(np.float32)
        if self.train:
            output = [rgbn, rgb_pad]
        else:
            output = [rgbn, gt]
        if self.return_idx:
            output.append(np.array([index], dtype=int))
        return output
