import time
import torch
from torch import nn, optim

import sys
#sys.path.append("..")
import utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(cin, cout, num_residuals, first_block=False):
    if first_block:
        assert cin == cout
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(cin, cout, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(cout,cout,use_1x1conv=False, stride=1))

    return nn.Sequential(*blk)

def resnet():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = resnet_block(64, 64, 2, first_block=True)
    b3 = resnet_block(64, 128, 2)
    b4 = resnet_block(128, 256, 2)
    b5 = resnet_block(256, 512, 2)
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        d2l.GlobalAvgPool2d(),
                        d2l.FlattenLayer(),
                        nn.Linear(512, 10))
    return net

batch_size=256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

net = resnet()
print(net)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
