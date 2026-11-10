import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import pywt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Multi_feature_Fusion_Module(nn.Module):
    def __init__(self, channel, is_last=False):
        super(Multi_feature_Fusion_Module, self).__init__()

        self.is_last = is_last

        self.SE_Block = SE_Block(channel, is_dis=False)
        self.Up = nn.Upsample(scale_factor=2)

        self.Conv1_1 = ConvLayer(channel * 2, channel, 3, 1, True)
        self.Conv1_2 = ConvLayer(channel, channel, 3, 1, True)

        # Only for final MFM (channel=16 → output=1)
        self.Conv4_2 = ConvLayer(16, 1, 3, 1, True)

    def forward(self, Fused_Image, FeatureMap_Ir, Feats_Ir, FeatureMap_Vis, Feats_Vis):

        add_Ir = self.Conv1_1(FeatureMap_Ir) + Feats_Ir
        add_Vis = self.Conv1_1(FeatureMap_Vis) + Feats_Vis

        Fused_Image = self.Conv1_1(Fused_Image)

        w = self.SE_Block(Fused_Image)

        add_all = w * add_Ir + (1 - w) * add_Vis
        out = self.Conv1_2(self.Up(add_all))

        # final MFM produces 1-channel fused output
        if self.is_last:
            out = self.Conv4_2(self.Up(add_all))

        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv2d(out)
        if self.use_Prelu:
            out = self.act(out)
        return out

class SE_Block(nn.Module):
    def __init__(self, channel, is_dis=False):
        super(SE_Block, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.is_dis = is_dis  # discriminator uses ONLY weights

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)     # (B,C)
        y = self.FC(y).view(b, c, 1, 1)    # (B,C,1,1)

        if self.is_dis:
            return y  # discriminator uses pure weights only

        return x * y  # generator uses weighted feature maps
    
class DWT(nn.Module):
    def __init__(self, in_channels):
        super(DWT, self).__init__()
        self.dwt = DWT_2D(in_channels=in_channels, wavename='haar')
        self.Conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        LL = self.Conv(LL)            # low-freq transform
        HF = LH + HL + HH             # high-freq combined
        return HF, LL

class DWT_2D(nn.Module):
    """
    2D Discrete Wavelet Transform (Haar) implemented via depthwise 2D convolution.
    Completely resolution-agnostic and 100% natively compatible with ONNX dynamic shapes.
    """
    def __init__(self, in_channels=16, wavename='haar'):
        super(DWT_2D, self).__init__()
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        low = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)
        high = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=torch.float32)

        # Pre-repeat for in_channels so weight is a CONSTANT buffer of shape (in_channels, 1, 2, 2)
        self.register_buffer('w_LL', torch.outer(low, low).view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1), persistent=False)
        self.register_buffer('w_LH', torch.outer(low, high).view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1), persistent=False)
        self.register_buffer('w_HL', torch.outer(high, low).view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1), persistent=False)
        self.register_buffer('w_HH', torch.outer(high, high).view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1), persistent=False)
        self.in_channels = in_channels

    def forward(self, x):
        LL = F.conv2d(x, self.w_LL, stride=2, groups=self.in_channels)
        LH = F.conv2d(x, self.w_LH, stride=2, groups=self.in_channels)
        HL = F.conv2d(x, self.w_HL, stride=2, groups=self.in_channels)
        HH = F.conv2d(x, self.w_HH, stride=2, groups=self.in_channels)

        return LL, LH, HL, HH

class DWTFunction_2D(Function):

    @staticmethod
    def forward(ctx, x, low_h, low_w, high_h, high_w):

        ctx.save_for_backward(low_h, low_w, high_h, high_w)

        L = torch.matmul(low_h, x)
        H = torch.matmul(high_h, x)

        LL = torch.matmul(L, low_w)
        LH = torch.matmul(L, high_w)
        HL = torch.matmul(H, low_w)
        HH = torch.matmul(H, high_w)

        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, gLL, gLH, gHL, gHH):
        low_h, low_w, high_h, high_w = ctx.saved_tensors

        gL = torch.matmul(gLL, low_w.t()) + torch.matmul(gLH, high_w.t())
        gH = torch.matmul(gHL, low_w.t()) + torch.matmul(gHH, high_w.t())

        gx = torch.matmul(low_h.t(), gL) + torch.matmul(high_h.t(), gH)
        return gx, None, None, None, None