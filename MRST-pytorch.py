import time
import torch
from torch import nn, optim

import sys
#sys.path.append("..")
import utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

def scale(x, y):#x:(B*C*T*H*W);y=(B*C)
    x = x.permute(2, 3, 4, 0, 1)
    y = y.squeeze()
    z = x * y
    z = z.permute(3, 4, 0, 1, 2)
    return z

class GlobalAvgPool3d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class MRST(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_conv_x=False, stride=1):
        super(MRST, self).__init__()
        self.bottleneck_1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1)
        self.conv_s = nn.Conv3d(mid_channels, mid_channels, kernel_size=(1,3,3), padding=(0,1,1), stride=stride)
        self.conv_t = nn.Conv3d(mid_channels, mid_channels, kernel_size=(3,1,1), padding=(1,0,0), stride=stride)
        self.conv_ss = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv_st = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv_ts = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv_tt = nn.Conv3d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.bottleneck_2 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0, stride=1)
        if use_conv_x:
            self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv_x = None

        self.bn_s = nn.BatchNorm3d(mid_channels)
        self.bn_t = nn.BatchNorm3d(mid_channels)

    def forward(self, X):
        X_ = F.relu(self.bottleneck_1(X))
        S = F.relu(self.bn_s(self.conv_s(X_)))
        T = F.relu(self.bn_t(self.conv_t(X_)))
        S_m = self.conv_ss(S) + self.conv_ts(T)
        T_m = self.conv_ts(S) + self.conv_tt(T)
        s_a = torch.sigmoid(F.avg_pool3d(S_m, kernel_size= S_m.size()[2:]))
        t_a = torch.sigmoid(F.avg_pool3d(T_m, kernel_size= T_m.size()[2:]))
        S = scale(S, s_a)
        T = scale(T, t_a)
        Y = self.bottleneck_2(F.relu(S + T))
        if self.conv_x:
            X = self.conv_x(X)
        return F.relu(Y + X)

def resnet_block(in_channels, mid_channels, out_channels, num_residuals):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            blk.append(MRST(in_channels, mid_channels, out_channels, use_conv_x=True, stride=2))
        else:
            blk.append(MRST(out_channels, mid_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)))

net.add_module("resnet_block1", resnet_block(64, 64, 256, 3))
net.add_module("resnet_block2", resnet_block(256, 128, 512, 4))
net.add_module("resnet_block3", resnet_block(512, 256, 1024, 6))
net.add_module("resnet_block4", resnet_block(1024, 512, 2048, 3))

net.add_module("global_avg_pool", GlobalAvgPool3d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(2048, 10)))

X = torch.rand((2, 1, 16, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)