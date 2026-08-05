import torch
import torch.nn as nn

from Modules import (
    DWT,
    ConvLayer,
    Multi_feature_Fusion_Module,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  GENERATOR (DWT-based Encoder–Decoder Fusion Network)
class Generator_DWT(nn.Module):
    def __init__(self):
        super(Generator_DWT, self).__init__()

        # Encoder
        self.ConvBlock_1 = nn.Sequential(
            ConvLayer(1, 16, 3, 1, True),
            ConvLayer(16, 16, 3, 1, True),
        )
        self.ConvBlock_2 = nn.Sequential(
            ConvLayer(16, 32, 3, 1, True),
            ConvLayer(32, 32, 3, 2, True),
        )
        self.ConvBlock_3 = nn.Sequential(
            ConvLayer(32, 64, 3, 1, True),
            ConvLayer(64, 64, 3, 2, True),
        )
        self.ConvBlock_4 = nn.Sequential(
            ConvLayer(64, 128, 3, 1, True),
            ConvLayer(128, 128, 3, 2, True),
        )
        self.ConvBlock_5 = nn.Sequential(
            ConvLayer(128, 256, 3, 1, True),
            ConvLayer(256, 256, 3, 2, True),
        )

        # Wavelet (LL, HF using DWT)
        self.DWT_1 = DWT(16)
        self.DWT_2 = DWT(32)
        self.DWT_3 = DWT(64)
        self.DWT_4 = DWT(128)

        # Decoder (MFM)
        self.MFM_1 = Multi_feature_Fusion_Module(128)
        self.MFM_2 = Multi_feature_Fusion_Module(64)
        self.MFM_3 = Multi_feature_Fusion_Module(32)
        self.MFM_4 = Multi_feature_Fusion_Module(16, is_last=True)

    # FORWARD PASS
    def forward(self, Input_Ir, Input_Vis):
        # IR Encoder
        Ir_1 = self.ConvBlock_1(Input_Ir)
        FeatsHigh_Ir_1, LL1 = self.DWT_1(Ir_1)

        Ir_2 = self.ConvBlock_2(Ir_1)
        FeatsHigh_Ir_2, LL2 = self.DWT_2(Ir_2)

        Ir_3 = self.ConvBlock_3(Ir_2 + LL1)
        FeatsHigh_Ir_3, LL3 = self.DWT_3(Ir_3)

        Ir_4 = self.ConvBlock_4(Ir_3 + LL2)
        FeatsHigh_Ir_4, LL4 = self.DWT_4(Ir_4)

        Ir_5 = self.ConvBlock_5(Ir_4 + LL3)
        Ir_5 = Ir_5 + LL4

        # VIS Encoder
        Vis_1 = self.ConvBlock_1(Input_Vis)
        FeatsHigh_Vis_1, LL1 = self.DWT_1(Vis_1)

        Vis_2 = self.ConvBlock_2(Vis_1)
        FeatsHigh_Vis_2, LL2 = self.DWT_2(Vis_2)

        Vis_3 = self.ConvBlock_3(Vis_2 + LL1)
        FeatsHigh_Vis_3, LL3 = self.DWT_3(Vis_3)

        Vis_4 = self.ConvBlock_4(Vis_3 + LL2)
        FeatsHigh_Vis_4, LL4 = self.DWT_4(Vis_4)

        Vis_5 = self.ConvBlock_5(Vis_4 + LL3)
        Vis_5 = Vis_5 + LL4

        # Fusion (element-wise addition)
        FusedImage = Ir_5 + Vis_5

        # Decoder (MFM blocks)
        R1 = self.MFM_1(FusedImage, Ir_5, FeatsHigh_Ir_4, Vis_5, FeatsHigh_Vis_4)
        R2 = self.MFM_2(R1, Ir_4, FeatsHigh_Ir_3, Vis_4, FeatsHigh_Vis_3)
        R3 = self.MFM_3(R2, Ir_3, FeatsHigh_Ir_2, Vis_3, FeatsHigh_Vis_2)
        R4 = self.MFM_4(R3, Ir_2, FeatsHigh_Ir_1, Vis_2, FeatsHigh_Vis_1)

        return R4