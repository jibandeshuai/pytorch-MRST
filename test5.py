import time
import torch
from torch import nn, optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range (num_convs):
        if (i==0):
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    return nn.Sequential(*blk)



def fc_block(fc_in, fc_hidden, fc_out):
    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(fc_in, fc_hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden, fc_hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden, fc_out)
    )
    return net


def vgg_net(conv_arch, fc_in, fc_hidden, fc_out):
    net = nn.Sequential()
    for i, (num, cin, cout) in enumerate(conv_arch):
        net.add_module("vgg_block_"+str(i), vgg_block(num, cin, cout))

    net.add_module("fc", fc_block(fc_in, fc_hidden, fc_out))
    return net

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_in = 512*7*7
fc_hidden = 4096
fc_out = 10

net = vgg_net(conv_arch, fc_in, fc_hidden, fc_out)
print(net)

batch_size = 64
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
