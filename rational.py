# coding=utf-8
# Copyright: Michael Zhu
"""  . """
import random

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from torch import nn


def _get_xps(z, len_numerator, len_denominator):
    xps = []
    xps.append(z)
    for _ in range(max(len_numerator, len_denominator) - 2):
        xps.append(xps[-1].mul(z))
    xps.insert(0, torch.ones_like(z))
    return torch.stack(xps, 1)


def Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)


class Rational(nn.Module):

    def __init__(self, approx_func):
        super(Rational, self).__init__()

        if approx_func == "gelu":
            init_w_numerator = [
                -0.0004221071456647063,
                0.49999955516606254,
                0.40270535451115536,
                0.07366976895222031,
                -0.012954788054484537,
                -0.0037414002583076983
            ]
            init_w_denominator = [
                4.9585381197087913e-05,
                0.1472977407631199,
                1.1645825701440633e-05,
                -0.007483871514842074
            ]
        else:

            init_w_numerator = [
                0.033897129202224346,
                0.4999985439606278,
                1.6701363611130988,
                1.9901021632350815,
                0.9413089613384323,
                0.1509133373584318
            ]
            init_w_denominator = [
                -2.1040152094202414e-05,
                3.980247851167207,
                -3.166344237241501e-05,
                0.30183382300945066
            ]

        self.numerator = nn.Parameter(torch.FloatTensor(init_w_numerator).to(torch.bfloat16),
                                      requires_grad=True)
        self.denominator = nn.Parameter(torch.FloatTensor(init_w_denominator).to(torch.bfloat16),
                                        requires_grad=True)

        self.activation_function = Rational_PYTORCH_B_F

    def forward(self, x):
        return self.activation_function(x, self.numerator, self.denominator,
                                        self.training)