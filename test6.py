import time
import torch
from torch import nn, optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(nin_in, nin_out, kernel_size, stride, padding):
    net = nn.Sequential(
        nn.Conv2d(nin_in, nin_out, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(nin_out, nin_out, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(nin_out, nin_out, kernel_size=1),
        nn.ReLU()
    )
    return net

import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])



def nin_net():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        GlobalAvgPool2d(),
        d2l.FlattenLayer()
    )
    return net

net = nin_net()
print(net)

batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)