'''SimpleNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'v1': [8, 16, 32,  64],
    'v2': [16, 32, 64,  64],
    'v3': [16, 32, 64, 128]
}


class SimpleBlock(nn.Module):
    def __init__(self, in_planes, planes, bias=False, stride=1, merged=False):
        super(SimpleBlock, self).__init__()
        self.merged = merged
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, bias=bias, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(planes)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

    def forward(self, x):
        out = self.conv1(x)
        
        if not self.merged:
            out = self.bn1(out)
            
        out = F.relu(out)
        if self.stride == 1:
            out = F.max_pool2d(out, 2)
        return out


class SimpleNet112(nn.Module):
    def __init__(self, pretrained=None, in_channels = 3):
        super(SimpleNet112, self).__init__()
        self.in_channels = in_channels
        # 16, 32, 64, 128
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(16),
            F.relu(out),
            F.max_pool2d(out, 2),

            nn.Conv2d(16, 32, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            F.relu(out),
            F.max_pool2d(out, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(64),
            F.relu(out),
            F.max_pool2d(out, 2),

            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            F.relu(out),
            F.max_pool2d(out, 2),
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=5),
            out = out.view(out.size(0), -1))
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
