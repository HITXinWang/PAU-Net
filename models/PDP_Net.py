import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math
from torch.autograd import Variable
from PDP_Net import LWAConv2D, LWAReverseConv2D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 上采样module
class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, patch_size):
        m = []
        if (scale & (scale - 1)) == 0:  #
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PDPNet(nn.Module):
	def __init__(self, embed_dim, patch_size, sr_scale, num_blocks):
		super(PDPNet, self).__init__()
		self.embed_dim = embed_dim
		self.patch_size = patch_size
		self.patch_h = patch_size[0]
		self.patch_w = patch_size[1]
		self.sr_scale = sr_scale
		self.num_blocks = num_blocks

		# modules
		self.lrelu = nn.LeakyReLU(inplace=True)
		self.input = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

		self.bn1 = nn.BatchNorm2d(self.embed_dim)
		self.lrelu1 = nn.LeakyReLU(inplace=True)
		self.conv1 = LWAConv2D(in_channels=self.embed_dim, out_channels=self.embed_dim//2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, patch_h=self.patch_h, patch_w=self.patch_w)

		self.bn2 = nn.BatchNorm2d(self.embed_dim//2)
		self.lrelu2 = nn.LeakyReLU(inplace=True)
		self.conv2 = LWAConv2D(in_channels=self.embed_dim//2, out_channels=self.embed_dim, kernel_size=3, stride=1,
							   padding=1, dilation=1, groups=1, bias=True, patch_h=self.patch_h, patch_w=self.patch_w)

		self.upsample = Upsample(self.sr_scale, self.embed_dim, self.patch_size)

		# restore ODIs
		self.lrelu_up = nn.LeakyReLU(inplace=True)
		self.up_enhance1 = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim//4, kernel_size=3, stride=1, padding=1,
							  dilation=1, groups=1, bias=True)
		self.up_enhance2 = nn.Conv2d(in_channels=self.embed_dim//4, out_channels=self.embed_dim, kernel_size=3,
									 stride=1, padding=1, dilation=1, groups=1, bias=True)

		self.lrelu_last = nn.LeakyReLU(inplace=True)
		self.last = nn.Conv2d(in_channels=self.embed_dim, out_channels=3, kernel_size=3, stride=1, padding=1,
								dilation=1, groups=1, bias=True)
		# weights initialization
		self.apply(self._init_weights)

	def forward(self, x):
		x = self.input(self.lrelu(x))
		inputs = x
		for _ in range(self.num_blocks):
			x = self.conv1(self.lrelu1(self.bn1(x)))
			x = self.conv2(self.lrelu2(self.bn2(x)))
			x = x + inputs
		x = self.upsample(x)
		re = x
		for _ in range(3):
			x = self.up_enhance1(x)
			x = self.up_enhance2(x)
		x = x + re
		x = self.last(self.lrelu_last(x))
		return x

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1.0)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)