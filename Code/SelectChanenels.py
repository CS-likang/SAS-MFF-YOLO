# -*-coding utf-8 -*-
# @Time :2020/6/9 16:52
# @Author : change zhou heng 
# To become better
import math

import torch
import torch.nn as nn
from expriment.test import visual, count
import torch.nn.functional as F


class SC_block(nn.Module):
    def __init__(self, out_channels, ratio=2, relu=True):
        super(SC_block, self).__init__()
        self.out_channels = out_channels

        ## TODO 实现将多个特征层进行concat拼接
        # self.down_channel = nn.ModuleList()
        # for channel in in_channels:
        #     self.down_channel.append(nn.Conv2d(channel, out_channels, 1, 1));

        ## TODO 选取重要通道
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(out_channels, out_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_channels // ratio, out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        self.init_channels = math.ceil(out_channels / ratio)

        ## TODO 生成所有通道
        new_channels = self.init_channels * (ratio - 1)
        self.cheap_operation = nn.Sequential(  # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(self.init_channels, new_channels, 3, 1, 3 // 2, groups=self.init_channels, bias=False),
            # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.down_channel = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        count([x], "count_x1")
        # visual([x],"vis_x")
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # avg_out = self.avg_pool(x)
        # max_out = self.max_pool(x)
        out = avg_out + max_out
        scales = self.sigmoid(out)
        _, positive_channels = torch.sort(scales, dim=1, descending=True)  ## 从大到小排序，返回是最大值对应的索引
        # positive_channel = positive_channels[:, :self.init_channels, :, :].contiguous().view(-1)
        negative_channel = positive_channels[:, self.init_channels:, :, :].contiguous().view(-1)
        channels = torch.arange(1,x.size(1)+1).bool()
        # channels[positive_channel] = True
        channels[negative_channel] = False

        _x = x.view(-1, x.size(2), x.size(3))
        x1 = _x[channels, :, :]
        x1 = x1.view(x.size(0), self.init_channels, x.size(2), x.size(3))
        # count([x1], "count_x1")
        # visual([x1],"vis_x1")

        x2 = self.cheap_operation(x1)
        out = torch.cat([x * scales, x1, x2], dim=1)  # torch.cat: 在给定维度上对输入的张量序列进行连接操作
        # 将黄色部分和红色部分在通道上进行拼接
        out = self.down_channel(out)
        # visual([out])
        # count([out],"count_out")
        return out  # 输出Fig2中的output；由于向上取整，可以会导致通道数大于self.out


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup  # oup：论文Fig2(a)中output的通道数
        init_channels = math.ceil(oup / ratio)  # init_channels: 在论文Fig2(b)中,黄色部分的通道数
        # ceil函数：向上取整，
        # ratio：在论文Fig2(b)中，output通道数与黄色部分通道数的比值
        new_channels = init_channels * (ratio - 1)  # new_channels: 在论文Fig2(b)中，output红色部分的通道数

        self.primary_conv = nn.Sequential(  # 输入所用的普通的卷积运算，生成黄色部分
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            # 1//2=0
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(  # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)  # torch.cat: 在给定维度上对输入的张量序列进行连接操作
        # 将黄色部分和红色部分在通道上进行拼接
        return out[:, :self.oup, :, :]  # 输出Fig2中的output；由于向上取整，可以会导致通道数大于self.out


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, ratio=16, num_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(num_channels, num_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // ratio, num_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        self.init_channels = math.ceil(num_channels / ratio)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        scales = self.sigmoid(out)
        positive_scales = torch.sort(scales, dim=1)[1].ge(self.init_channels).view(-1)
        out_x = x[:, positive_scales, :, :]

        select_channel_nums = scales.sort()

        return x + x * scales
