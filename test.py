#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
import datasets
from PIL import Image
import itertools


def test():
    net.eval()
    for mode, testset, testloader in zip(modes, testsets, testloaders):
        env_name = config.name + '_' + str(config.resume_seed)
        if isinstance(mode, (list, tuple)):
            save_path = os.path.join('results', env_name, *[str(i) for i in mode])
        else:
            save_path = os.path.join('results', env_name, str(mode))
        txt_path = save_path + '.txt'
        os.makedirs(save_path, exist_ok=True)
        Avg = AverageMeter()
        for batch_idx, (rgbn, rgb, idx) in enumerate(testloader):
            rgbn, rgb = rgbn.cuda(), rgb.cuda()
            h, w = rgb.shape[2:]
            fh, fw = rgbn.shape[2:]
            wl = (fw - w) // 2
            hl = (fh - h) // 2
            with torch.no_grad():
                pred = net(rgbn)
                pred[pred < 0] = 0
                pred[pred > 1] = 1
                pred = pred[:, :, hl:hl + h, wl:wl + w]
                prec = metric(pred, rgb)
            pred = pred[0].cpu().numpy().transpose(1, 2, 0)
            pred = (255 * pred).round().astype(np.uint8)
            idx = idx.item()
            name = testset.names[idx]
            filename = os.path.join(save_path, name)
            Img = Image.fromarray(pred)
            Img.save(filename)
            Avg.update(prec.item(), rgb.size(0))
        with open(txt_path, 'w') as f:
            f.write('{} = {}'.format(config.metric, Avg.avg))


if __name__ == '__main__':
    # config_name = 'DB.yaml'
    # config_name = 'DL.yaml'
    config_name = 'DLN.yaml'

    modes = [
        'test_lmdb/kodak',
        'test_lmdb/mcm',
        'test_lmdb/hdrvdp',
        'test_lmdb/moire',
    ]
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    from utils import *

    transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    if 'noise' in params:
        noise = params.pop('noise')
        modes = list(itertools.product(modes, noise))
        testsets = [dataset(**params, mode=mode, train=False, noise=noise, transform=transform, return_idx=True) for
                    mode, noise in modes]
    else:
        testsets = [dataset(**params, mode=mode, train=False, transform=transform, return_idx=True) for mode in modes]
    testloaders = [torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
                   for testset in testsets]
    net = init_net(config)
    metric = init_metric(config)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    net.cuda()
    net = torch.nn.DataParallel(net)
    metric = torch.nn.DataParallel(metric)
    net = resume_state(config, net)
    test()
