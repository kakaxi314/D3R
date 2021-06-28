#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    create_lmdb.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import os
import numpy as np
import glob
from PIL import Image
import lmdb
import time
import pyarrow as pa


def pull_RGB(path):
    img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
    return img


def dumps_pyarrow(obj):
    return pa.serialize(obj).to_buffer()


def create_lmdb(paths):
    write_frequency = 1000
    LMDB_MAP_SIZE = 1 << 40  # MODIFY
    for path in paths:
        input_path = path
        output_path = path.split('/')
        output_path[2] += '_lmdb'
        output_path = str.join('/', output_path)
        os.makedirs(output_path, exist_ok=True)
        print(output_path)
        db = lmdb.open(output_path, map_size=LMDB_MAP_SIZE, readonly=False,
                       meminit=False, map_async=True)
        images = list(sorted(glob.iglob(input_path + "/**/*.png", recursive=True))) + \
                 list(sorted(glob.iglob(input_path + "/**/*.tif", recursive=True)))
        print(len(images))
        images = np.array(images)
        image_order = np.random.permutation(len(images))
        images = images[image_order]
        images = list(images)
        t0 = time.time()
        txn = db.begin(write=True)
        names = []
        for idx in range(len(images)):
            if idx % write_frequency == 0:
                print(idx)
                t1 = time.time()
                print(t1 - t0)
                t0 = t1
            img_data = pull_RGB(images[idx])
            img_name = os.path.split(images[idx])[-1]
            names.append(img_name)
            txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((img_data,)))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(images)))
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(len(images))]
        with db.begin(write=True) as txn:
            txn.put(b'__names__', dumps_pyarrow(names))
            txn.put(b'__keys__', dumps_pyarrow(keys))
            txn.put(b'__len__', dumps_pyarrow(len(keys)))
        print("Finished")
        db.sync()
        db.close()


if __name__ == '__main__':
    paths = [
        'datas/color/train',
        'datas/color/val',
        'datas/color/test/kodak',
        'datas/color/test/mcm',
        'datas/color/test/hdrvdp',
        'datas/color/test/moire',
    ]
    create_lmdb(paths)
