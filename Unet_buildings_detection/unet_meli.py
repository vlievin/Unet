"""
U-net model from https://arxiv.org/abs/1505.04597 implemented with residual blocks, batch norm and dropout 
couple with distance net module, multi task learning described in https://arxiv.org/pdf/1709.05932.pdf

"""

import torch
from torch import nn
from torch.nn import functional as F



class inconv(nn.Module):
    """
    initial convolution for the U-Net model
    """
    def __init__(self, in_ch, out_ch,width_kernel,dropout,batch_norm):
        super(inconv, self).__init__()
        self.conv = make_residual_layer(ResidualBlock, in_ch, out_ch, 1, width_kernel,dropout=dropout,batch_norm=batch_norm)

    def forward(self, x):
        x = self.conv(x)
        return x
class outconv(nn.Module):
    """
    last convolution for the U-Net (Fully connected layer)
    """
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
def conv3x3(in_channels, out_channels, width_kernel,stride=1):
    """
    3x3 convolution
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=width_kernel, 
                     stride=stride, padding=int((width_kernel-1)/2), bias=False)

class ResidualBlock(nn.Module):
    """
    Residual Block with two convolution, dropout and batch normalization
    dropout set according to https://arxiv.org/pdf/1605.07146.pdf
    """
    def __init__(self, in_channels, out_channels, width_kernel,stride=1, downsample=None, dropout=0,batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.conv1 = conv3x3(in_channels, out_channels,width_kernel, stride)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        if self.dropout > 0:
            self.drop = nn.Dropout2d(self.dropout)
        self.conv2 = conv3x3(out_channels, out_channels,width_kernel)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        if self.dropout > 0:
            out = self.drop(out)
        #out = self.relu(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
def make_residual_layer(block,in_channels, out_channels, blocks, width_kernel,stride=1, dropout=0,batch_norm=False):
    """
    make a layer from intermediate blocks
    """
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = nn.Sequential(
            conv3x3(in_channels, out_channels, width_kernel,stride=stride),
            nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(block(in_channels, out_channels,width_kernel, stride, downsample, dropout=dropout, batch_norm=batch_norm))
    in_channels = out_channels
    for i in range(1, blocks):
        layers.append(block(out_channels, out_channels,width_kernel, dropout=dropout, batch_norm=batch_norm))
    return nn.Sequential(*layers)


class down_residual(nn.Module):
    """
    U-Net down block using on residual block
    """
    def __init__(self, in_ch, out_ch,width_kernel, dropout=0,batch_norm=False):
        super(down_residual, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            make_residual_layer(ResidualBlock, in_ch, out_ch, 1, width_kernel,dropout=dropout,batch_norm=batch_norm)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
class up_residual(nn.Module):
    """
    U-Net up block using one residual block
    """
    def __init__(self, in_ch, out_ch,width_kernel, dropout=0,batch_norm=False):
        super(up_residual, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = make_residual_layer(ResidualBlock, in_ch, out_ch,1, width_kernel,dropout=dropout,batch_norm=batch_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, diffX // 2,diffY // 2, diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    """
    U-Net model with custom number of layers, dropout and batch normalization
    """
    def __init__(self, in_channels, out_channels, depth = 5, n_features_zero = 64, width_kernel=3,dropout=0,batch_norm=True,distance_net=False,bins=15):
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=3)
            out_channels (int): number of output channels (n_classes)
            depth (int): number of down/up layers
            n_features (int): number of initial features
            dropout (float): float in [0,1]: dropout probability
            batch_norm (bool): use batch normalization or not
            distance_net (bool): add distance Net module
            bins (int): number of bins of the distance map
            
        """
        super(UNet, self).__init__()
        n_features = n_features_zero
        self.inc = inconv(in_channels,n_features,width_kernel,dropout=dropout,batch_norm=batch_norm)
        # DOWN
        self.downs = torch.nn.ModuleList()
        for k in range(depth-1):
            d = down_residual(n_features, 2*n_features,width_kernel, dropout=dropout,batch_norm=batch_norm)
            n_features = 2 * n_features
            self.downs += [d]
        self.downs += [down_residual(n_features, n_features,width_kernel, dropout=dropout,batch_norm=batch_norm)]
        # UP
        self.ups = torch.nn.ModuleList()
        for k in range(depth):
            u = up_residual(2*n_features, n_features//2,width_kernel, dropout=dropout,batch_norm=batch_norm)
            n_features = n_features // 2
            self.ups += [u]
        self.outc = outconv(n_features, out_channels)
        self.distance_net=distance_net
        if self.distance_net:
            self.outc2= outconv(n_features, bins)
        
    def forward(self, x):
        x = self.inc(x)
        bridges = []
        for d in self.downs:
            bridges += [x]
            x = d(x)
 
        for k,u in enumerate(self.ups):
            x = u(x,bridges[len(bridges)-1-k])
        if self.distance_net:
            x_seg=self.outc(x)
            x_dist=torch.cat((x, x_seg), 1)
            x_dist = self.outc2(x)
            return x_dist,x_seg
        else:
            x = self.outc(x)
            return x
        
    
    def debug(self, x):
        x = self.inc(x)
        bridges = []
        downs = []
        ups = []
        for d in self.downs:
            bridges += [x]
            x = d(x)
            downs.append(x)
        for k,u in enumerate(self.ups):
            x = u(x,bridges[len(bridges)-1-k])
            ups.append(x)
        x = self.outc(x)
        return x, downs, ups
