import time
import torch
from torch import nn, optim

import sys
#sys.path.append("..")
import utils as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

class Inception_block(nn.Module):
    def __init__(self, cin, c1, c2, c3, c4):
        super(Inception_block, self).__init__()
        self.p1 = nn.Conv2d(cin, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(cin, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding = 1)
        self.p3_1 = nn.Conv2d(cin, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(cin, c4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1(x)
        p1 = F.relu(p1)
        #print(p1.size())
        p2 = self.p2_2(F.relu(self.p2_1(x)))
        p2 = F.relu(p2)
        #print(p2.size())
        p3 = self.p3_2(F.relu(self.p3_1(x)))
        p3 = F.relu(p3)
        #print(p3.size())
        p4 = self.p4_2(F.relu(self.p4_1(x)))
        p4 = F.relu(p4)
        #print(p4.size())
        out = torch.cat((p1, p2, p3, p4), dim=1)
        return out

def Inception_net():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b3 = nn.Sequential(
        Inception_block(192, 64, [96, 128], [16, 32], 32),
        Inception_block(256, 128, [128, 192], [32, 96], 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b4 = nn.Sequential(
        Inception_block(480, 192, (96, 208), (16, 48), 64),
        Inception_block(512, 160, (112, 224), (24, 64), 64),
        Inception_block(512, 128, (128, 256), (24, 64), 64),
        Inception_block(512, 112, (144, 288), (32, 64), 64),
        Inception_block(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b5 = nn.Sequential(Inception_block(832, 256, (160, 320), (32, 128), 128),
                       Inception_block(832, 384, (192, 384), (48, 128), 128),
                       d2l.GlobalAvgPool2d())

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        d2l.FlattenLayer(), nn.Linear(1024, 10))
    return net

batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

net = Inception_net()
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
