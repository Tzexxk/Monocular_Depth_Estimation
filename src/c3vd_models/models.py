import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb, depth, highlight=None):
        x = torch.cat([rgb, depth], dim=1)
        return depth + self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 32)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = ConvBlock(192, 64)
        self.dec1 = ConvBlock(96, 32)
        self.output_conv = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, rgb, depth, highlight=None):
        x = torch.cat([rgb, depth], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = F.interpolate(
            b, size=e2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(
            d2, size=e1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return depth + self.output_conv(d1)


class GatedMultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = ConvBlock(3, 16)
        self.depth_encoder = ConvBlock(1, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(33, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),
            nn.Sigmoid()
        )
        self.enc1 = ConvBlock(16, 32)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = ConvBlock(192, 64)
        self.dec1 = ConvBlock(96, 32)
        self.output_conv = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, rgb, depth, highlight=None):
        if highlight is None:
            highlight = torch.zeros_like(depth)

        rf = self.rgb_encoder(rgb)
        df = self.depth_encoder(depth)

        gate = self.gate(torch.cat([rf, df, highlight], dim=1))
        fused = df + gate * rf

        e1 = self.enc1(fused)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = F.interpolate(
            b, size=e2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return depth + self.output_conv(d1)


MODELS = {"simple_cnn": SimpleCNN, "multiscale_b": MultiScaleCNN, "gated_c0": GatedMultiScaleCNN, "highlight_c1": GatedMultiScaleCNN}
