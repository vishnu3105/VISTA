"""
OSNet-x1.0 built from actual weight keys+shapes. Loads 100%.

Key facts:
- gate.fc1/fc2 are Conv2d(1x1) not Linear — shape [4,64,1,1]
- _LightConv: conv1=pointwise[c,c,1,1], conv2=depthwise[c,1,3,3]
- Transition pool: conv2.2 = Sequential(ConvLayer, AvgPool2d)
- fc = Sequential(Linear[512,512], BN1d[512])
- mid channels: conv2=64, conv3=96, conv4=128
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _LightConv(nn.Module):
    # conv1=[c,c,1,1] pointwise, conv2=[c,1,3,3] depthwise
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn    = nn.BatchNorm2d(c)
    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class ChannelGate(nn.Module):
    # fc1/fc2 are Conv2d(1x1) — confirmed by shape [c//16, c, 1, 1]
    def __init__(self, c):
        super().__init__()
        mid = c // 16
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)
    def forward(self, x):
        s = x.mean([2, 3], keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, mid):
        super().__init__()
        self.conv1  = ConvLayer(in_c, mid, 1)
        self.conv2a = _LightConv(mid)
        self.conv2b = nn.Sequential(_LightConv(mid), _LightConv(mid))
        self.conv2c = nn.Sequential(_LightConv(mid), _LightConv(mid), _LightConv(mid))
        self.conv2d = nn.Sequential(_LightConv(mid), _LightConv(mid),
                                    _LightConv(mid), _LightConv(mid))
        self.gate       = ChannelGate(mid)
        self.conv3      = ConvLayer(mid, out_c, 1)
        self.downsample = ConvLayer(in_c, out_c, 1) if in_c != out_c else None
    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)+self.conv2b(x1)+self.conv2c(x1)+self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample: identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, 7, s=2, p=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            OSBlock(64,  256, mid=64),
            OSBlock(256, 256, mid=64),
            nn.Sequential(ConvLayer(256, 256, 1), nn.AvgPool2d(2, stride=2)),
        )
        self.conv3 = nn.Sequential(
            OSBlock(256, 384, mid=96),
            OSBlock(384, 384, mid=96),
            nn.Sequential(ConvLayer(384, 384, 1), nn.AvgPool2d(2, stride=2)),
        )
        self.conv4 = nn.Sequential(
            OSBlock(384, 512, mid=128),
            OSBlock(512, 512, mid=128),
        )
        self.conv5          = ConvLayer(512, 512, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc             = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512))

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)