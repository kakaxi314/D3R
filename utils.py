#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    utils.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import os
import torch
import random
import numpy as np
import augs
import models
import datasets
import criteria

__all__ = [
    'AverageMeter',
    'init_seed',
    'init_aug',
    'init_dataset',
    'init_cuda',
    'init_net',
    'init_loss',
    'init_metric',
    'init_optim',
    'init_lr_scheduler',
    'save_state',
    'resume_state',
]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def config_param(model):
    param_groups = []
    other_params = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            g = {'params': [param], 'weight_decay': 0.0}
            param_groups.append(g)
        elif 'cfa' in name:
            g = {'params': [param], 'weight_decay': 0.0}
            param_groups.append(g)
        else:
            other_params.append(param)
    param_groups.append({'params': other_params})
    return param_groups


def save_state(config, model):
    print('==> Saving model ...')
    env_name = config.name + '_' + str(config.manual_seed)
    save_path = os.path.join('checkpoints', env_name)
    os.makedirs(save_path, exist_ok=True)
    model_state_dict = model.state_dict()
    state_dict = {
        'net': model_state_dict,
    }
    torch.save(state_dict, os.path.join(save_path, 'result.pth'))


def resume_state(config, model):
    env_name = config.name + '_' + str(config.resume_seed)
    cp_path = os.path.join('checkpoints', env_name, 'result.pth')
    resume_model = torch.load(cp_path)['net']
    model.load_state_dict(resume_model, strict=True)
    return model


def init_seed(config):
    if config.manual_seed == 0:
        config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    torch.initial_seed()
    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)


def init_net(config):
    return getattr(models, config.model)()


def init_loss(config):
    return getattr(criteria, config.loss)()


def init_metric(config):
    return getattr(criteria, config.metric)()


def init_aug(aug_config):
    transform = []
    for x in aug_config:
        print(x)
        if type(x) == str:
            transform.append(getattr(augs, x)())
        else:
            key, params = x.popitem()
            transform.append(getattr(augs, key)(**params))
    return augs.Compose(transform)


def init_dataset(config):
    train_transform = init_aug(config.train_aug_configs)
    test_transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    trainset = dataset(**params, mode='train_lmdb', transform=train_transform)
    testset = dataset(**params, mode='val_lmdb', transform=test_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              num_workers=config.num_workers, shuffle=False, drop_last=True,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                             num_workers=config.num_workers, shuffle=False, drop_last=True,
                                             pin_memory=True)
    print('num_train = {}, num_test = {}'.format(len(trainset), len(testset)))
    return trainloader, testloader


def init_cuda(net, criterion, metric):
    torch.cuda.empty_cache()
    net.cuda()
    criterion.cuda()
    metric.cuda()
    net = torch.nn.DataParallel(net)
    criterion = torch.nn.DataParallel(criterion)
    metric = torch.nn.DataParallel(metric)
    torch.backends.cudnn.benchmark = True
    return net, criterion, metric


def init_optim(config, net):
    key, params = config.optim_config.popitem()
    return getattr(torch.optim, key)(config_param(net), **params)


def init_lr_scheduler(config, optimizer):
    key, params = config.lr_config.popitem()
    return getattr(torch.optim.lr_scheduler, key)(optimizer, **params)
