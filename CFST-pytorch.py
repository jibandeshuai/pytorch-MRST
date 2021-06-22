import time
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

def scale(x, y):#x:(B*C*T*H*W);y=(B*C*1*1*1)
    x = x.permute(2, 3, 4, 0, 1)
    y = y.squeeze()
    z = x * y
    z = z.permute(3, 4, 0, 1, 2)
    return z

def new_scale(x, y):#x:(B*C*T*H*W);y=(B)
    x = x.permute(1, 2, 3, 4, 0)
    z = x * y
    z = z.permute(4, 0, 1, 2, 3)
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

class CFST(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv_x=False, tmp_stride=1, spa_stride=1):
        super(CFST, self).__init__()
        self.conv_s1, self.conv_s2, self.conv_t1, self.conv_t2 = [], [], [], []
        self.conv_ss1, self.conv_st1, self.conv_ts1, self.conv_tt1 = [], [], [], []
        self.conv_ss2, self.conv_st2, self.conv_ts2, self.conv_tt2 = [], [], [], []
        for i in range(16):
            self.convs1_i = nn.Conv3d(in_channels//16, out_channels//16, kernel_size=(1,3,3), padding=(0,1,1),
                                      stride=(tmp_stride, spa_stride, spa_stride))
            self.convs2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                      stride=(1, 1, 1))
            self.convt1_i = nn.Conv3d(in_channels//16, out_channels//16, kernel_size=(3,1,1), padding=(1,0,0),
                                      stride=(tmp_stride, spa_stride, spa_stride))
            self.convt2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                                      stride=(1, 1, 1))
            self.conv_ss1_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_st1_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_ts1_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_tt1_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_ss2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_st2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_ts2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_tt2_i = nn.Conv3d(out_channels // 16, out_channels // 16, kernel_size=1, padding=0)
            self.conv_s1 += [self.convs1_i]
            self.conv_t1 += [self.convt1_i]
            self.conv_s2 += [self.convs2_i]
            self.conv_t2 += [self.convt2_i]
            self.conv_ss1 += [self.conv_ss1_i]
            self.conv_st1 += [self.conv_st1_i]
            self.conv_ts1 += [self.conv_ts1_i]
            self.conv_tt1 += [self.conv_tt1_i]
            self.conv_ss2 += [self.conv_ss2_i]
            self.conv_st2 += [self.conv_st2_i]
            self.conv_ts2 += [self.conv_ts2_i]
            self.conv_tt2 += [self.conv_tt2_i]
        self.CFR_1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1,1,1), padding=0, stride=1)
        self.CFR_2 = nn.Conv3d(in_channels*2, in_channels, kernel_size=1, padding=0, stride=1)
        if use_conv_x:
            self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(tmp_stride, spa_stride, spa_stride))
        else:
            self.conv_x = None
        self.bn_s1, self.bn_t1 = [], []
        self.bn_s2, self.bn_t2 = [], []
        for i in range(16):
            self.bn_s1_i = nn.BatchNorm3d(out_channels//16)
            self.bn_t1_i = nn.BatchNorm3d(out_channels//16)
            self.bn_s2_i = nn.BatchNorm3d(out_channels // 16)
            self.bn_t2_i = nn.BatchNorm3d(out_channels // 16)
            self.bn_s1 +=[self.bn_s1_i]
            self.bn_t1 +=[self.bn_t1_i]
            self.bn_s2 += [self.bn_s2_i]
            self.bn_t2 += [self.bn_t2_i]
        
        for i in range(16):
            self.conv_s1[i] = self.conv_s1[i].cuda()
            self.conv_t1[i] = self.conv_t1[i].cuda()
            self.conv_s2[i] = self.conv_s2[i].cuda()
            self.conv_t2[i] = self.conv_t2[i].cuda()
            self.conv_ss1[i] = self.conv_ss1[i].cuda()
            self.conv_st1[i] = self.conv_st1[i].cuda()
            self.conv_ts1[i] = self.conv_ts1[i].cuda()
            self.conv_tt1[i] = self.conv_tt1[i].cuda()
            self.conv_ss2[i] = self.conv_ss2[i].cuda()
            self.conv_st2[i] = self.conv_st2[i].cuda()
            self.conv_ts2[i] = self.conv_ts2[i].cuda()
            self.conv_tt2[i] = self.conv_tt2[i].cuda()
            self.bn_s1[i] = self.bn_s1[i].cuda()
            self.bn_t1[i] = self.bn_t1[i].cuda()
            self.bn_s2[i] = self.bn_s2[i].cuda()
            self.bn_t2[i] = self.bn_t2[i].cuda()
            
    def forward(self, X):
        #CFR
        BX = []
        for i in range(16):
            BX += [X[:, i * (X.size()[1] // 16):(i + 1) * (X.size()[1] // 16), :, :, :]]
        X_ = F.relu(self.CFR_1(X))
        XX,a,P = [],[],[]
        weight = []
        for i in range(16):
            XX += [X_[:,i*(X_.size()[1]//16):(i+1)*(X_.size()[1]//16),:,:,:]]
            a_i = F.avg_pool3d(XX[i], kernel_size= XX[i].size()[2:])
            a_i = a_i.squeeze()
            a += [a_i]

        for i in range(16):
            cor = []
            for j in range(16):
                if j==i:
                    continue
                else:
                    tmp = torch.sum((a[j]-a[i])**2, dim = 1)
                    tmp = tmp.view(a[i].size()[0], -1)
                    cor.append(tmp)
            res = cor[0]
            print()
            for k in range(1, len(cor)):
                res = torch.cat((res, cor[k]), 1)
            res = F.softmax(res.float(), dim=1)
            res = res.permute(1, 0)
            weight += [res]
        for i in range(16):
            p_i = 0
            for j in range(16):
                if j==i:
                    continue
                elif j<i:
                    p_i += new_scale(XX[j], weight[i][j])
                else:
                    p_i += new_scale(XX[j], weight[i][j-1])

            P+=[p_i]
        for i in range(16):
            XX[i] = torch.cat((XX[i], P[i]), dim=1)
        X_ = XX[0]
        for i in range(1, 16):
            X_ = torch.cat((X_, XX[i]), dim=1)
        X_ = F.relu(self.CFR_2(X_))
        XX_ = []
        #STC-Unit
        for i in range(16):
            XX_ += [X_[:,i*(X_.size()[1]//16):(i+1)*(X_.size()[1]//16),:,:,:]]
        Y1, Y2 = [], []
        for i in range(16):
            S1 = F.relu(self.bn_s1[i](self.conv_s1[i](XX_[i])))
            T1 = F.relu(self.bn_t1[i](self.conv_t1[i](XX_[i])))
            S1_ = self.conv_ss1[i](S1) + self.conv_ts1[i](T1)
            T1_ = self.conv_st1[i](S1) + self.conv_tt1[i](T1)
            Y1_i = F.relu(S1_+T1_)
            Y1 += [Y1_i]
        for i in range(16):
            S2 = F.relu(self.bn_s2[i](self.conv_s2[i](Y1[i])))
            T2 = F.relu(self.bn_t2[i](self.conv_t2[i](Y1[i])))
            S2_ = self.conv_ss1[i](S2) + self.conv_ts1[i](T2)
            T2_ = self.conv_st1[i](S2) + self.conv_tt1[i](T2)
            Y2_i = F.relu(S2_ + T2_)
            Y2 += [Y2_i]
        Y = Y2[0]
        for i in range(1, 16):
            Y = torch.cat((Y, Y2[i]), dim=1)
        if self.conv_x:
            X = self.conv_x(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            #不想损失tmporal_size， 将tmp_stride设为1
            blk.append(CFST(in_channels, out_channels, use_conv_x=True, tmp_stride=2, spa_stride=2))
        else:
            blk.append(CFST(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)))

net.add_module("resnet_block1", resnet_block(64, 64, 3, first_block = True))
net.add_module("resnet_block2", resnet_block(64, 128, 4))
net.add_module("resnet_block3", resnet_block(128, 256, 6))
net.add_module("resnet_block4", resnet_block(256, 512, 3))

net.add_module("global_avg_pool", GlobalAvgPool3d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 101)))

X = torch.rand((2, 1, 16, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
