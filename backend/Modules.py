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
        self.dwt = DWT_2D(wavename='haar')
        self.Conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        LL = self.Conv(LL)            # low-freq transform
        HF = LH + HL + HH             # high-freq combined
        return HF, LL

class DWT_2D(nn.Module):

    """
    Input  : (N, C, H, W)
    Output : LL, LH, HL, HH (each N,C,H/2,W/2)
    """

    def __init__(self, wavename='haar'):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.low = wavelet.rec_lo
        self.high = wavelet.rec_hi
        self.L = len(self.low)
        self.half = self.L // 2
        self._filter_cache = {}

    def _get_filters(self, H, W, device):
        key = (H, W, str(device))
        if key in self._filter_cache:
            return self._filter_cache[key]

        L = max(H, W)
        out = math.floor(L / 2)

        # build low filter matrices
        mat_low = np.zeros((out, L + self.L - 2))
        index = 0
        for i in range(out):
            for j in range(self.L):
                mat_low[i, index + j] = self.low[j]
            index += 2
        mat_low = mat_low[:, (self.half - 1):(-(self.half - 1)) or None]

        # build high filter matrices
        mat_high = np.zeros((L - out, L + self.L - 2))
        index = 0
        for i in range(L - out):
            for j in range(self.L):
                mat_high[i, index + j] = self.high[j]
            index += 2
        mat_high = mat_high[:, (self.half - 1):(-(self.half - 1)) or None]

        mat_low_h = torch.tensor(mat_low[:math.floor(H / 2), :H], dtype=torch.float32, device=device)
        mat_low_w = torch.tensor(mat_low[:math.floor(W / 2), :W], dtype=torch.float32, device=device).t()

        mat_high_h = torch.tensor(mat_high[:H - math.floor(H / 2), :H], dtype=torch.float32, device=device)
        mat_high_w = torch.tensor(mat_high[:W - math.floor(W / 2), :W], dtype=torch.float32, device=device).t()

        self._filter_cache[key] = (mat_low_h, mat_low_w, mat_high_h, mat_high_w)
        return self._filter_cache[key]

    def forward(self, x):
        N, C, H, W = x.size()
        mat_low_h, mat_low_w, mat_high_h, mat_high_w = self._get_filters(H, W, x.device)

        LL, LH, HL, HH = DWTFunction_2D.apply(
            x,
            mat_low_h, mat_low_w,
            mat_high_h, mat_high_w
        )
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