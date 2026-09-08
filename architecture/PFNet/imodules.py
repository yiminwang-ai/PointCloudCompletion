#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from improved_codes.conv import Conv
from improved_codes.transformer import TransformerLayer


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SimConvlayer(nn.Module):
    def __init__(self, point_scales):
        super(SimConvlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
        self.sim=SimAM()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.sim(x)
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x

class VeryLightConvlayer(nn.Module):
    def __init__(self, point_scales):
        super(VeryLightConvlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 32, (1, 3))
        self.conv2 = torch.nn.Conv2d(32, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.sim=SimAM()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.sim(x)
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_64 = torch.squeeze(self.maxpool(x_64), 2)
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        L = [x_512, x_256, x_128,x_64]
        x = torch.cat(L, 1)
        return x
class GhostConvlayer(nn.Module):
    def __init__(self, point_scales):
        super(GhostConvlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = GhostConv(64, 64, 1)
        self.conv3 = GhostConv(64, 128, 1)
        self.conv4 = GhostConv(128, 256, 1)
        self.conv5 = GhostConv(256, 512, 1)
        self.conv6 = GhostConv(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x_128 = self.conv3(x)
        x_256 = self.conv4(x_128)
        x_512 = self.conv5(x_256)
        x_1024 = self.conv6(x_512)
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class VITlayer(nn.Module):
    def __init__(self, point_scales):
        super(VITlayer, self).__init__()
        self.point_scales = point_scales
        self.down1 = TransformerBlock(1, 64, 4, 1)
        #self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.down2 = TransformerBlock(64, 128, 4, 1)
        #self.conv2 = torch.nn.Conv2d(64, 64, 1)
        #self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.down3 = TransformerBlock(128, 256, 4, 1)
        self.down4 = TransformerBlock(256, 512, 4, 1)
        self.down5 = TransformerBlock(512, 1024, 4, 1)

        # self.conv4 = torch.nn.Conv2d(128, 256, 1)
        # self.conv5 = torch.nn.Conv2d(256, 512, 1)
        # self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)

        self.sim = SimAM()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.down1(x)
        x_128 = self.down2(x)
        x_256 = self.down3(x_128)
        x_512 = self.down4(x_256)
        x_1024 = self.down5(x_512)
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class SeperableConv(nn.Module):
    def __init__(self):
        super(SeperableConv, self).__init__()


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


if __name__ == "__main__":
    input = torch.randn(8, 1, 2048, 3)  # 64，2048，3
    sim = SimAM()
    out = sim.forward(input)
    print(out.shape)
