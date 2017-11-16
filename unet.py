"""
Unet with residual blocks implemented from https://arxiv.org/abs/1505.04597

"""

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
def make_residual_layer(block,in_channels, out_channels, blocks, stride=1):
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    in_channels = out_channels
    for i in range(1, blocks):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)


class down_residual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_residual, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            make_residual_layer(ResidualBlock, in_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
class up_residual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_residual, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = make_residual_layer(ResidualBlock, in_ch, out_ch, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        padX = x1.size()[2] - x2.size()[2]
        padY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (padX // 2, padX // 2, padY // 2, padY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = make_residual_layer(ResidualBlock, in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Residual_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Residual_UNet, self).__init__()
        self.inc = inconv(in_channels,64)
        self.down1 = down_residual(64, 128)
        self.down2 = down_residual(128, 256)
        self.down3 = down_residual(256, 512)
        self.down4 = down_residual(512, 1024)
        self.down5 = down_residual(1024, 1024)
        self.up5 = up_residual(2048, 512)
        self.up4 = up_residual(1024, 256)
        self.up3 = up_residual(512, 128)
        self.up2 = up_residual(256, 64)
        self.up1 = up_residual(128, 64)
        self.outc = outconv(64, out_channels)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up5(x6, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        x = x.transpose(1,2)
        x = x.transpose(2,3)
        return x
