import time
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

def scale(x, y):#x:(B*C*T*H*W);y=(B*C)
    x = x.permute(2, 3, 4, 0, 1)
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


class MSTI(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_conv_x=False, tmp_stride=1, spa_stride=1):
        super(MSTI, self).__init__()
        self.bottleneck_1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1)
        self.conv_s, self.conv_t = [], []
        self.conv_ss1, self.conv_ss, self.conv_ts = [], [], []
        self.conv_tt1, self.conv_tt, self.conv_st = [], [], []
        self.fc_s, self.fc_t = [], []
        for i in range(4):
            if i==0:
                self.convs_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         stride=(tmp_stride, spa_stride, spa_stride))
                self.convt_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         stride=(tmp_stride, spa_stride, spa_stride))
            else:
                self.convs_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                         stride=(tmp_stride, spa_stride, spa_stride))
                self.convt_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                                         stride=(tmp_stride, spa_stride, spa_stride))
            self.conv_ss_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
            self.conv_ts_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
            self.conv_tt_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
            self.conv_st_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
            self.fc_s_i = nn.Linear(mid_channels // 4, mid_channels // 4)
            self.fc_t_i = nn.Linear(mid_channels // 4, mid_channels // 4)
            self.conv_s += [self.convs_i]
            self.conv_t += [self.convt_i]
            self.conv_ss += [self.conv_ss_i]
            self.conv_ts += [self.conv_ts_i]
            self.conv_tt += [self.conv_tt_i]
            self.conv_st += [self.conv_st_i]
            self.fc_s += [self.fc_s_i]
            self.fc_t += [self.fc_t_i]
            if i>=1:
                self.conv_ss1_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
                self.conv_tt1_i = nn.Conv3d(mid_channels // 4, mid_channels // 4, kernel_size=1, padding=0)
                self.conv_ss1 += [self.conv_ss1_i]
                self.conv_tt1 += [self.conv_tt1_i]
        self.bottleneck_2 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0, stride=1)
        if use_conv_x:
            self.add = False
            self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                    stride=(tmp_stride, spa_stride, spa_stride))
        else:
            self.add = True
            self.conv_x = None
        self.bn_s, self.bn_t = [], []
        for i in range(4):
            self.bn_s_i = nn.BatchNorm3d(out_channels // 4)
            self.bn_t_i = nn.BatchNorm3d(out_channels // 4)
            self.bn_s += [self.bn_s_i]
            self.bn_t += [self.bn_t_i]
        for i in range(4):
            self.conv_s[i] = self.conv_s[i].cuda()
            self.conv_t[i] = self.conv_t[i].cuda()
            self.conv_ss[i] = self.conv_ss[i].cuda()
            self.conv_st[i] = self.conv_st[i].cuda()
            self.conv_ts[i] = self.conv_ts[i].cuda()
            self.conv_tt[i] = self.conv_tt[i].cuda()
            self.fc_s[i] = self.fc_s[i].cuda()
            self.fc_t[i] = self.fc_t[i].cuda()
            self.bn_s[i] = self.bn_s[i].cuda()
            self.bn_t[i] = self.bn_t[i].cuda()

        for i in range(3):
            self.conv_ss1[i] = self.conv_ss1[i].cuda()
            self.conv_tt1[i] = self.conv_tt1[i].cuda()



    def forward(self, X, ):
        X_ = F.relu(self.bottleneck_1(X))
        XX = []
        for i in range(4):
            XX += [X_[:,i*(X_.size()[1]//4):(i+1)*(X_.size()[1]//4),:,:,:]]
        S, T = [], []
        for i in range(4):
            if i==0:
                S_i = F.relu(self.bn_s[i](self.conv_s[i](XX[i])))
                T_i = F.relu(self.bn_t[i](self.conv_t[i](XX[i])))
                S += [S_i]
                T += [T_i]
            else:
                if self.add:
                    S_i = F.relu(self.bn_s[i](self.conv_s[i](XX[i] + S[-1])))
                    T_i = F.relu(self.bn_t[i](self.conv_t[i](XX[i] + T[-1])))
                else:
                    S_i = F.relu(self.bn_s[i](self.conv_s[i](XX[i])))
                    T_i = F.relu(self.bn_t[i](self.conv_t[i](XX[i])))
                S += [S_i]
                T += [T_i]
        S_m, T_m = [], []
        for i in range(4):
            if i==0:
                S_m_i = self.conv_ss[i](S[i]) + self.conv_ts[i](T[i])
                T_m_i = self.conv_tt[i](T[i]) + self.conv_st[i](S[i])
                S_m += [S_m_i]
                T_m += [T_m_i]
            else:
                S_m_i = self.conv_ss1[i-1](S[i-1]) + self.conv_ss[i](S[i]) + self.conv_ts[i](T[i])
                T_m_i = self.conv_tt1[i-1](T[i-1]) + self.conv_tt[i](T[i]) + self.conv_st[i](S[i])
                S_m += [S_m_i]
                T_m += [T_m_i]
        new_S, new_T = [], []
        for i in range(4):
            s_a_i = torch.sigmoid(self.fc_s[i](F.avg_pool3d(S_m[i], kernel_size=S_m[i].size()[2:]).squeeze()))
            t_a_i = torch.sigmoid(self.fc_t[i](F.avg_pool3d(T_m[i], kernel_size=T_m[i].size()[2:]).squeeze()))
            new_S_i = scale(S[i], s_a_i)
            new_T_i = scale(T[i], t_a_i)
            new_S += [new_S_i]
            new_T += [new_T_i]
        S_out = new_S[0]
        for i in range(1, 4):
            S_out = torch.cat((S_out, new_S[i]), dim=1)
        T_out = new_T[0]
        for i in range(1, 4):
            T_out = torch.cat((T_out, new_T[i]), dim=1)
        Y = self.bottleneck_2(F.relu(S_out + T_out))
        if self.conv_x:
            X = self.conv_x(X)
        return F.relu(Y + X)

def resnet_block(in_channels, mid_channels, out_channels, num_residuals, first_block = False):
    blk = []
    if first_block:
        for i in range(num_residuals):
            if i == 0:
                # 不想损失tmporal_size， 将tmp_stride设为1
                blk.append(MSTI(in_channels, mid_channels, out_channels, use_conv_x=True, tmp_stride=2, spa_stride=2))
            else:
                blk.append(MSTI(out_channels, mid_channels, out_channels))
        return nn.Sequential(*blk)
    else:
        for i in range(num_residuals):
            if i == 0:
                # 不想损失tmporal_size， 将tmp_stride设为1
                blk.append(MSTI(in_channels, mid_channels, out_channels, use_conv_x=True, tmp_stride=1, spa_stride=2))
            else:
                blk.append(MSTI(out_channels, mid_channels, out_channels))
        return nn.Sequential(*blk)


net = nn.Sequential(
        nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)))

net.add_module("resnet_block1", resnet_block(64, 64, 256, 3, True))
net.add_module("resnet_block2", resnet_block(256, 128, 512, 4))
net.add_module("resnet_block3", resnet_block(512, 256, 1024, 6))
net.add_module("resnet_block4", resnet_block(1024, 512, 2048, 3))

net.add_module("global_avg_pool", GlobalAvgPool3d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(2048, 101)))

X = torch.rand((2, 1, 16, 224, 224))
#查看网络输出
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)




