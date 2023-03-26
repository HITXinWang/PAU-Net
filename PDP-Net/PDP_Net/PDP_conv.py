import torch
from torch import nn
from util.ws_util import getWeightRec
import numpy as np


class LWAConv2D(nn.Module):
    '''  LWAConv2D is the implementation of PDP convolution.
    Note that
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, patch_h=256, patch_w=64):
        super(LWAConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ph = ((self.patch_h - 1) * self.stride - self.patch_h + self.kernel_size) // 2
        self.pw = ((self.patch_w - 1) * self.stride - self.patch_w + self.kernel_size) // 2
        self.weightRec = np.sqrt(np.sqrt(getWeightRec(0, self.patch_h, self.patch_w)))
        self.cov = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                             stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
                             bias=self.bias)

    def forward(self, x):
        with torch.no_grad():
            self.patchWeight = torch.Tensor(self.weightRec).cuda()
        self.patchWeight = self.patchWeight.type_as(x)
        x = x * self.patchWeight
        x = self.cov(x)
        return x

