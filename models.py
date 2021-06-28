#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    models.py
# @Project:     D3R
# @Author:      jie
# @Time:        2021/4/1 4:06 PM

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import PSC

__all__ = [
    'DB',
    'DL',
]


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, stride=1, padding=1, act=True):
        super().__init__()
        self.pad = padding
        if padding:
            self.pad = nn.ReplicationPad2d(padding)
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=0, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=0, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        if act:
            self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        if self.pad:
            out = self.pad(x)
        out = self.conv(out)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Basic2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, norm_layer=norm_layer)
        self.conv2 = Basic2d(planes, planes, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, act=False)
        self.downsample = downsample
        self.stride = stride
        self.act = act
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        if self.act:
            out = self.relu(out)
        return out


class solver_nl(nn.Module):

    def __init__(self, pattern_size=4, N=4, ks=5, lamb=1e-2):
        super(solver_nl, self).__init__()
        self.pattern_size = pattern_size
        self.N = N
        self.ks = ks
        self.lamb = nn.Parameter(lamb * torch.eye(N, dtype=torch.float32).view(1, 1, N, N))
        self.lamb.requires_grad = False
        self.CA_weight = nn.Parameter(torch.ones([N * N, 1, ks, ks], dtype=torch.float32))
        self.CA_weight.requires_grad = False
        self.CB_weight = nn.Parameter(torch.ones([N, 1, ks, ks], dtype=torch.float32))
        self.CB_weight.requires_grad = False

    def forward(self, dictionary, raw, cfa):
        b, c, h, w = dictionary.shape
        dict_sf = self.unshuffle_d(dictionary)
        cfa_sf = self.unshuffle_c(cfa)
        Ao = torch.matmul(cfa_sf, dict_sf)
        Aij = Ao.transpose(-1, -2).matmul(Ao).permute(0, 2, 3, 1).view(b, self.N * self.N, h, w)
        Ao = Ao.transpose(-1, -3).view(b, self.N, h, w)
        A_star = F.conv2d(Aij, self.CA_weight, padding=self.ks // 2, groups=self.N * self.N)
        A_star = A_star.permute(0, 2, 3, 1).contiguous().view(b, h * w, self.N, self.N)
        A_star = A_star + self.lamb
        # A_inverse = torch.inverse(A_star.view(-1, self.N, self.N)).view(b, h * w, self.N, self.N)
        A_inverse = self.batchedInv(A_star.view(-1, self.N, self.N)).view(b, h * w, self.N, self.N)
        Atb = Ao * raw
        Atb = F.conv2d(Atb, self.CB_weight, padding=self.ks // 2, groups=self.N)
        Atb = Atb.permute(0, 2, 3, 1).contiguous().view(b, h * w, self.N, 1)
        X = A_inverse.matmul(Atb)
        final = torch.matmul(dict_sf, X)
        final = final.transpose(1, 2).view(b, 3, h, w)
        return final

    def batchedInv(self, batchedTensor):
        if batchedTensor.shape[0] >= 256 * 256 - 1:
            temp = []
            for t in torch.split(batchedTensor, 256 * 256 - 1):
                temp.append(torch.inverse(t))
            return torch.cat(temp)
        else:
            return torch.inverse(batchedTensor)

    def unshuffle_d(self, img):
        b, c, h, w = img.shape
        output = img.contiguous().view(b, c, h * w)
        output = output.permute(0, 2, 1).contiguous().view(b, h * w, 3, self.N)
        return output

    def unshuffle_c(self, img):
        b, c, h, w = img.shape
        output = img.contiguous().view(b, c, h * w)
        output = output.permute(0, 2, 1).contiguous().view(b, h * w, 1, 3)
        return output


def truncate_init(cfa, kernel_size=2):
    n = kernel_size * kernel_size * 3
    data = truncated_normal_(n, mean=0, std=math.sqrt(1.3 * 2. / n))
    data = data.float()
    data = data.view(1, 3, kernel_size, kernel_size)
    cfa.cfa.data = data.view_as(cfa.cfa.data)
    return cfa


def bayer_init(cfa):
    data = np.zeros((3, 2, 2), dtype=np.float32)
    data[0, 0, 0] = 1
    data[1, 0, 1] = 1
    data[1, 1, 0] = 1
    data[2, 1, 1] = 1
    data = torch.from_numpy(data)
    cfa.cfa.data = data.view_as(cfa.cfa.data)
    return cfa


def truncated_normal_(num, mean=0, std=1):
    lower = -2 * std
    upper = 2 * std
    X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
    samples = X.rvs(num)
    output = torch.from_numpy(samples)
    return output


class mosaic_cfa_bias(nn.Module):
    """
    the mosaic process
    """

    def __init__(self, kernel_size=2, requires_grad=False):
        super(mosaic_cfa_bias, self).__init__()
        self.kernel_size = kernel_size
        self.cfa = nn.Parameter(torch.zeros([1, 3, kernel_size, kernel_size], dtype=torch.float32))
        self.cfa.requires_grad = requires_grad
        self.bias = nn.Parameter(torch.zeros([1, 1, kernel_size, kernel_size], dtype=torch.float32))

    def forward(self, img):
        b, c, h, w = img.shape
        cfa = self.cfa.repeat(1, 1, h // self.kernel_size, w // self.kernel_size)
        bias = self.bias.repeat(1, 1, h // self.kernel_size, w // self.kernel_size)
        img = img * cfa
        img = torch.sum(img, dim=1, keepdim=True)
        return img, cfa, bias


class mosaic_learn_cfa_bias(nn.Module):
    """
    the mosaic process
    """

    def __init__(self, kernel_size=2, requires_grad=True):
        super(mosaic_learn_cfa_bias, self).__init__()
        self.kernel_size = kernel_size
        self.cfa = nn.Parameter(torch.zeros([1, 3, kernel_size, kernel_size], dtype=torch.float32))
        self.soft_max = nn.Softmax(dim=1)
        self.cfa.requires_grad = requires_grad
        self.bias = nn.Parameter(torch.zeros([1, 1, kernel_size, kernel_size], dtype=torch.float32))

    def forward(self, img):
        b, c, h, w = img.shape
        cfa = self.soft_max(self.cfa)
        cfa = cfa.repeat(1, 1, h // self.kernel_size, w // self.kernel_size)
        bias = self.bias.repeat(1, 1, h // self.kernel_size, w // self.kernel_size)
        img = img * cfa
        img = torch.sum(img, dim=1, keepdim=True)
        return img, cfa, bias


def get_cfa_bias(cfa='bayer', pattern_size=4):
    if cfa == 'bayer':
        cfa = mosaic_cfa_bias(requires_grad=False)
        cfa = bayer_init(cfa)
    elif cfa == 'learn':
        cfa = mosaic_learn_cfa_bias(kernel_size=pattern_size, requires_grad=True)
        cfa = truncate_init(cfa, kernel_size=pattern_size)
    else:
        raise ValueError("unknown cfa")
    return cfa


class pa_conv2d_F(Function):
    @staticmethod
    def forward(ctx, *inputs):
        pattern_size = inputs[-1]
        input = inputs[0]
        weights = inputs[1:1 + pattern_size * pattern_size]
        biases = inputs[1 + pattern_size * pattern_size:1 + pattern_size * pattern_size * 2]
        stride, dilation, groups, benchmark, deterministic = inputs[1 + pattern_size * pattern_size * 2:-1]
        ctx.save_for_backward(input, *weights)
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.benchmark = benchmark
        ctx.deterministic = deterministic
        ctx.pattern_size = pattern_size
        padding = (0, 0)
        ctx.padding = padding
        outputs = []
        b, c, h, w = input.shape
        for i in range(pattern_size):
            for j in range(pattern_size):
                outputs.append(
                    PSC.Conv_F(input[:, :, i:h - pattern_size + i + 1, j:w - pattern_size + j + 1],
                               weights[i * pattern_size + j], biases[i * pattern_size + j], padding, stride,
                               dilation, groups, benchmark, deterministic)
                )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        input = inputs[0]
        weights = inputs[1:]
        b, c, h, w = input.shape
        grad_inputs = torch.zeros_like(input).type_as(input)
        grad_weights = []
        grad_biases = []
        for i in range(ctx.pattern_size):
            for j in range(ctx.pattern_size):
                grad_input, grad_weight = PSC.Conv_B(
                    input[:, :, i:h - ctx.pattern_size + i + 1, j:w - ctx.pattern_size + j + 1],
                    grad_outputs[i * ctx.pattern_size + j],
                    weights[i * ctx.pattern_size + j], ctx.padding, ctx.stride,
                    ctx.dilation, ctx.groups, ctx.benchmark, ctx.deterministic,
                    torch.backends.cudnn.allow_tf32, [True, True])
                grad_bias = grad_outputs[i * ctx.pattern_size + j].sum([0, 2, 3])
                grad_inputs[:, :, i:h - ctx.pattern_size + i + 1, j:w - ctx.pattern_size + j + 1] += grad_input
                grad_weights.append(grad_weight)
                grad_biases.append(grad_bias)
        return (grad_inputs, *grad_weights, *grad_biases, None, None, None, None, None, None, None)


class pa_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, benchmark=True,
                 deterministic=False, pattern_size=4):
        super(pa_conv2d, self).__init__()
        self.stride = (stride * pattern_size, stride * pattern_size)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.benchmark = benchmark
        self.deterministic = deterministic
        self.pattern_size = pattern_size
        self.pixel_shuffle = nn.PixelShuffle(pattern_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.pad = nn.ReplicationPad2d(kernel_size // 2)
        self.weights = []
        self.biases = []
        for i in range(pattern_size):
            for j in range(pattern_size):
                weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
                bias = nn.Parameter(torch.Tensor(out_channels))
                self.register_parameter('weight_{i}_{j}'.format(i=i, j=j), weight)
                self.register_parameter('bias_{i}_{j}'.format(i=i, j=j), bias)
                self.weights.append(weight)
                self.biases.append(bias)
        self.reset_parameters()

    def forward(self, input):
        b, c, h, w = input.shape
        input = self.pad(input)
        weights = self.weights
        biases = self.biases
        outputs = pa_conv2d_F.apply(input, *weights, *biases, self.stride, self.dilation, self.groups,
                                    self.benchmark, self.deterministic, self.pattern_size)
        output = torch.stack(outputs, dim=2).view(b, -1, h // self.pattern_size, w // self.pattern_size)
        output = self.pixel_shuffle(output)
        return output

    def reset_parameters(self):
        for weight in self.weights:
            n = self.kernel_size * self.kernel_size * self.in_channels
            data = truncated_normal_(weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
            data = data.type_as(weight.data)
            weight.data = data.view_as(weight.data)
        for bias in self.biases:
            bias.data.zero_()


class PSC2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, benchmark=True,
                 deterministic=False, pattern_size=4):
        super().__init__()
        self.stride = (stride * pattern_size, stride * pattern_size)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.benchmark = benchmark
        self.deterministic = deterministic
        self.pattern_size = pattern_size
        self.pixel_shuffle = nn.PixelShuffle(pattern_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.pad = nn.ReplicationPad2d(kernel_size // 2)
        self.weights = []
        self.biases = []
        for i in range(pattern_size):
            for j in range(pattern_size):
                weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
                bias = nn.Parameter(torch.Tensor(out_channels))
                self.register_parameter('weight_{i}_{j}'.format(i=i, j=j), weight)
                self.register_parameter('bias_{i}_{j}'.format(i=i, j=j), bias)
                self.weights.append(weight)
                self.biases.append(bias)
        self.reset_parameters()

    def forward(self, input):
        b, c, h, w = input.shape
        input = self.pad(input)
        weights = []
        biases = []
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                weights.append(getattr(self, 'weight_{i}_{j}'.format(i=i, j=j)))
                biases.append(getattr(self, 'bias_{i}_{j}'.format(i=i, j=j)))
        outputs = pa_conv2d_F.apply(input, *weights, *biases, self.stride, self.dilation, self.groups,
                                    self.benchmark, self.deterministic, self.pattern_size)
        output = torch.stack(outputs, dim=2).view(b, -1, h // self.pattern_size, w // self.pattern_size)
        output = self.pixel_shuffle(output)
        return output

    def reset_parameters(self):
        for weight in self.weights:
            n = self.kernel_size * self.kernel_size * self.in_channels
            data = truncated_normal_(weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
            data = data.type_as(weight.data)
            weight.data = data.view_as(weight.data)
        for bias in self.biases:
            bias.data.zero_()


class bp_2d(nn.Module):
    def __init__(self, in_channels, out_channels, pattern_size=4, bn=None, relu=True, kernel_size=3):
        super(bp_2d, self).__init__()
        if bn == 'bn':
            self.conv = nn.Sequential(
                pa_conv2d(in_channels=in_channels, out_channels=out_channels, pattern_size=pattern_size,
                          kernel_size=kernel_size),
                nn.BatchNorm2d(num_features=out_channels),
            )
        elif bn is None:
            self.conv = nn.Sequential(
                pa_conv2d(in_channels=in_channels, out_channels=out_channels, pattern_size=pattern_size,
                          kernel_size=kernel_size),
            )
        else:
            raise ValueError("unknown bn")
        if relu:
            self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class position_conv(nn.Module):
    def __init__(self, pattern_size=2, bc=64, bn='bn'):
        super(position_conv, self).__init__()
        self.conv0 = bp_2d(1, bc, pattern_size=pattern_size, bn=bn, relu=True, kernel_size=5)

    def forward(self, input):
        output_c = self.conv0(input)
        return output_c


class UNet(nn.Module):

    def __init__(self, pattern_size=2, block=BasicBlock, layers=[2, 2, 2, 2], solver=solver_nl, bc=64,
                 norm_layer=nn.BatchNorm2d, cfa='bayer', N=4, ks=5, lamb=1e-2):
        super().__init__()
        self._norm_layer = norm_layer
        self.cfa = get_cfa_bias(cfa, pattern_size=pattern_size)
        self.pattern_size = pattern_size
        self.psc = position_conv(pattern_size=pattern_size, bc=bc)
        self.inplanes = bc
        self.layer0 = self._make_layer(block, bc, layers[0], stride=1)
        self.inplanes = bc
        self.layer1 = self._make_layer(block, bc * 2, layers[1], stride=2)
        self.inplanes = bc * 2
        self.layer2 = self._make_layer(block, bc * 4, layers[2], stride=2)
        self.inplanes = bc * 4
        self.layer3 = self._make_layer(block, bc * 8, layers[3], stride=2)
        self.layer0d = Basic2dTrans(bc * 2, bc, norm_layer)
        self.layer1d = Basic2dTrans(bc * 4, bc * 2, norm_layer)
        self.layer2d = Basic2dTrans(bc * 8, bc * 4, norm_layer)
        self.ref = block(bc * block.expansion, bc, stride=1, norm_layer=norm_layer)
        self.conv0 = Basic2d(bc * block.expansion, 3 * N, kernel_size=3, padding=1, norm_layer=norm_layer, act=False)
        self.solver = solver(pattern_size, N, ks=ks, lamb=lamb)
        self._initialize_weights()

    def forward(self, img):
        raw, cfa, bias = self.cfa(img)
        input_bias = raw + bias
        output = self.psc(input_bias)
        c0_img = self.layer0(output)
        c1_img = self.layer1(c0_img)
        c2_img = self.layer2(c1_img)
        c3_img = self.layer3(c2_img)
        dc2_img = self.layer2d(c3_img)
        c2 = c2_img + dc2_img
        dc1_img = self.layer1d(c2)
        c1 = c1_img + dc1_img
        dc0_img = self.layer0d(c1)
        c0 = c0_img + dc0_img
        ref = self.ref(c0)
        basis = self.conv0(ref)
        final = self.solver(basis, raw, cfa)
        return final

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def DB():
    """
    Demosaicing for Bayer Pattern CFA
    """
    return UNet(pattern_size=2, cfa='bayer', solver=solver_nl, bc=64, N=4, ks=5, lamb=1e-2)


def DL():
    """
    Demosaicing for Learned Pattern CFA
    """
    return UNet(pattern_size=4, cfa='learn', solver=solver_nl, bc=64, N=4, ks=5, lamb=1e-2)
