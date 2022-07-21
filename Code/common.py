# This file contains modules common to various models
from math import log

from utils.utils import *

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from dropblock import DropBlock2D


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.cv1 = SAS(c1, c_, 1, 1)
        # self.cv2 = SAS(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


"""Split-Attention"""


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                           groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class MutilSpatial(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=1, d=1, g=1, gamma=2, b=1):
        super(MutilSpatial, self).__init__()
        self.t = c1 // (int(abs((log(c1, 2) + b) / gamma)))
        self.num_groups = math.ceil(c1 / self.t)
        self.conv1 = Conv(2, 1, k=3, )
        self.conv2 = Conv(2, 1, k=7, )
        self.conv = nn.Conv2d(c1, c2, kernel_size=k)
        self.sigmoid = nn.Sigmoid()
        # TODO 通道特征之间存在周期性，考虑周期联系

    def forward(self, x):
        splits = torch.split(x, self.t, dim=1)
        squeezes = [
            torch.cat((torch.mean(split, dim=1).unsqueeze(dim=1), torch.max(split, dim=1)[0].unsqueeze(dim=1)), dim=1)
            for split in splits]
        # squeezes_sum = [torch.sum(split, dim=1).unsqueeze(dim=1) for split in splits]
        # attens_1 = [self.conv1(squeeze).view(squeeze.size(0), -1, squeeze.size(-2) * squeeze.size(-1)).permute(0, 2, 1)
        #             for squeeze in squeezes]  ## [b, w*h, c]
        # attens_2 = [self.conv2(squeeze).view(squeeze.size(0), -1, squeeze.size(-2) * squeeze.size(-1)) for squeeze in
        #             squeezes]  ## [b, c, w*h]
        # relation = [torch.bmm(atten1, atten2) for atten1, atten2 in
        #             zip(attens_1, attens_2)]  ## [b,w*h,c] X [b,c,w*h] = [b,w*h,w*h]
        # attens = [torch.softmax(rela, dim=-1) for rela in relation]  ## [b,w*h,w*h]
        # features = [torch.bmm(split.view(split.size(0), -1, split.size(-2) * split.size(-1)),
        #                       atten.permute(0, 2, 1)).reshape_as(split) for
        #             split, atten in
        #             zip(splits, attens)]

        attens_1 = [self.conv1(squeeze)
                    for squeeze in squeezes]  ## [b, w*h, c]
        attens_2 = [self.conv2(squeeze) for squeeze in
                    squeezes]  ## [b, c, w*h]
        features = [split * self.sigmoid(attens1) + split * self.sigmoid(attens2) for split, attens1, attens2 in
                    zip(splits, attens_1, attens_2)]
        end_features = torch.cat(features, dim=1)
        end_features = self.conv(end_features)
        return end_features;


class ECANet_Channel(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECANet_Channel, self).__init__()
        t = int(abs((log(in_channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        score = self.sigmoid(y)
        return score


class Shrink(nn.Module):
    def __init__(self, c1):
        super(Shrink, self).__init__()
        self.globalpooling = nn.AdaptiveAvgPool2d(1)
        self.full_1 = nn.Sequential(nn.Linear(c1, c1 // 8),
                                    nn.BatchNorm1d(c1 // 8),
                                    nn.ReLU())
        self.full_2 = nn.Sequential(nn.Linear(c1 // 8, c1),
                                    nn.Sigmoid())

    def forward(self, x):
        abs = torch.abs(x)
        mean_abs = self.globalpooling(abs)
        scores_tmp = self.full_2(self.full_1(mean_abs.view(mean_abs.size(0), -1)))
        scores_tmp = scores_tmp.view_as(mean_abs)
        scores = mean_abs * scores_tmp
        out_tmp = torch.mul(torch.sign(x), torch.max((abs - scores), 0)[0])
        out = x + out_tmp
        return out


class ContextDetail(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, compress_channel=8):
        super(ContextDetail, self).__init__()
        r = 2
        self.r = r
        tmp_channels_1 = c1 // r
        tmp_channels_2 = c1 - tmp_channels_1
        # self.conv = Conv(c1,)
        ## detials
        # self.conv_1 = nn.Conv2d(tmp_channels_1, c1, 1, 1, groups=g)
        self.detials_conv = nn.Conv2d(c1, tmp_channels_1, kernel_size=1, stride=1)
        self.conv_2 = Conv(tmp_channels_1, c2 // 2, g=g)

        ## encode
        self.encode_conv = nn.Conv2d(c1, tmp_channels_2, kernel_size=1, stride=1)
        self.encode1 = Conv(tmp_channels_2, c2 // 2, k=3, s=2, p=None, g=g)
        self.decode1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.encode2 = Conv(c2 // 2, c2 - (c2 // 2), k=3, s=2, p=None, g=g)
        self.decode2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.decode2 = nn.ConvTranspose2d(tmp_channels_2, c2, kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1, groups=g)
        ## decode
        # self.conv1x1 = nn.Conv2d(tmp_channels_2, tmp_channels_2 * r * r, kernel_size=1, stride=1,
        #                          padding=autopad(1, None), groups=g)
        # self.decode1 = nn.PixelShuffle(r)
        self.endconv = Conv(c2, c2, k, s, p, g)

        # TODO ASFF
        # self.weights_detail = Conv(c2 // 2, compress_channel)
        # self.weights_semantic = Conv( c2 // 2, compress_channel)
        # self.weights = nn.Conv2d(compress_channel * 2, 2, kernel_size=1, stride=1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # split1, split2 = torch.split(x, x.size(1) // self.r, dim=1)
        split1 = self.detials_conv(x)
        split2 = self.encode_conv(x)
        tmp_out_1 = self.conv_2(split1)

        _tmp = self.encode1(split2);
        _tmp = self.decode1(_tmp)
        # weights_detail = self.weights_detail(tmp_out_1)
        # weights_semantic = self.weights_semantic(_tmp)
        # weights = torch.cat((weights_detail, weights_semantic), dim=1)
        # weights = self.weights(weights)
        # weights = self.softmax(weights)
        # _tmp = _tmp * weights[:, 0:1, :, :] + tmp_out_1 * weights[:, 1:2, :, :]
        _tmp = self.encode2(_tmp)
        tmp_out_2 = self.decode2(_tmp)

        outs = torch.cat((tmp_out_1, tmp_out_2), dim=1)
        outs = self.endconv(outs)
        return outs


class ResShrink(nn.Module):
    def __init__(self, c1, c2):
        super(ResShrink, self).__init__()
        self.layer_1 = nn.Sequential(nn.BatchNorm2d(c1),
                                     nn.ReLU(),
                                     nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1))
        self.layer_2 = nn.Sequential(nn.BatchNorm2d(c2),
                                     nn.ReLU(),
                                     nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1))
        self.globalpooling = nn.AdaptiveAvgPool2d(1)
        self.full_1 = nn.Sequential(nn.Linear(c2, c2 // 8),
                                    nn.BatchNorm1d(c2 // 8),
                                    nn.ReLU())
        self.full_2 = nn.Sequential(nn.Linear(c2 // 8, c2),
                                    nn.Sigmoid())

    def forward(self, x):
        x_tmp = self.layer_2(self.layer_1(x))
        x_abs = torch.abs(x_tmp)
        x_abs_mean = self.globalpooling(x_abs)
        scores_tmp = self.full_2(self.full_1(x_abs_mean.view(x_abs_mean.size(0), -1)))
        scores_tmp = scores_tmp.view_as(x_abs_mean)
        scores = x_abs_mean * scores_tmp
        # sub = x_abs - scores
        out_tmp = torch.mul(torch.sign(x), torch.max((x_abs - scores), 0)[0])
        out = x + out_tmp
        return out


class ASFF(nn.Module):
    def __init__(self, c1, c2, c3, compress_channel=8):
        super(ASFF, self).__init__()
        self.weights_detail = Conv(c1, compress_channel)
        self.weights_semantic = Conv(c2, compress_channel)
        self.weights_residual = Conv(c3, compress_channel)
        self.weights = nn.Conv2d(compress_channel * 3, 3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        weights_detail = self.weights_detail(x1)
        weights_semantic = self.weights_semantic(x2)
        weights_residual = self.weights_residual(x3)
        weights = torch.cat((weights_detail, weights_semantic, weights_residual), dim=1)
        weights = self.weights(weights)
        weights = self.softmax(weights)
        out = torch.cat((x1 * weights[:, 0:1, :, :], x2 * weights[:, 1:2, :, :], x3 * weights[:, 2:3, :, :]), dim=1)
        return out


class ShrinkAttention(nn.Module):
    def __init__(self, c1):
        super(ShrinkAttention, self).__init__()
        self.atten1 = Conv(2, 1, k=3, )
        self.atten2 = nn.Sequential(Conv(2, 2, 3),
                                    Conv(2, 1, 3))
        self.shrink = ResShrink(c1, c1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.shrink(x)
        compress = torch.cat((torch.mean(x, dim=1).unsqueeze(dim=1), torch.max(x, dim=1)[0].unsqueeze(dim=1)), dim=1)
        tmp = self.atten1(compress) + self.atten2(compress)
        atten = self.sigmoid(tmp)
        return atten



class MFF(nn.Module):
    def __init__(self, c1, dimension=1):
        super(MFF, self).__init__()
        self.d = dimension
        self.eca = ECANet_Channel(c1)
        self.init_channels = c1 // 2
        self.cheap_operation = Conv(self.init_channels + 1, self.init_channels-1, 1)

    def forward(self, x):
        x2 = x[1]
        scales = self.eca(x2).view(x2.size(0), -1)
        _, positive_channels = torch.sort(scales, dim=1, descending=True)  ## 从大到小排序，返回是最大值对应的索引
        positive_channel = positive_channels[:, :self.init_channels].split(1)
        channel_tmp = torch.zeros(x2.size(0), x2.size(1)).bool().split(1)
        x_posi = torch.ones(x2.size(0), self.init_channels, x2.size(2), x2.size(3)).type_as(x2)
        x_nega = torch.ones(x2.size(0), x2.size(1) - self.init_channels, x2.size(2), x2.size(3)).type_as(x2)
        for i in range(x2.size(0)):
            channel_tmp[i][:, positive_channel[i].view(-1)] = True
            x_posi[i] = x2[i][channel_tmp[i].view(-1), :, :]
            x_nega[i] = x2[i][~channel_tmp[i].view(-1), :, :]
        tmp1 = torch.cat((x_posi, torch.mean(x_nega, dim=1).unsqueeze(dim=1)), dim=1)
        tmp2 = self.cheap_operation(tmp1)
        out = torch.cat([tmp1, tmp2], dim=1)
        x[1] = x2 + out
        return torch.cat(x, self.d)


class SAS(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(SAS, self).__init__()
        self.ch_1 = c1
        ## TODO 选取重要通道 0711
        self.atten_1 = Conv(2, 1, 7)  ##TODO 7.23 注意力
        self.eca = ECANet_Channel(c1)
        self.init_channels = c1 // 2
        ## TODO 通道特征分别计算
        r = 2
        self.r = r

        ## detials
        tmp_channels_1 = c1 // r
        tmp_channels_2 = c1 - tmp_channels_1
        self.conv_2 = Conv(tmp_channels_1, c2 // 2, g=g)
        ## TODO 0723 注意力
        self.atten_2 = Conv(2, 1, 3)
        ## encode
        self.encode1 = Conv(tmp_channels_2, tmp_channels_2, k=3, s=2, p=None, g=g)
        self.shrink1 = Shrink(tmp_channels_2)
        self.decode1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.encode2 = Conv(tmp_channels_2, c2 - (c2 // 2), k=3, s=2, p=None, g=g)
        self.shrink2 = Shrink(c2 - (c2 // 2))
        self.decode2 = nn.UpsamplingNearest2d(scale_factor=2)
        ## TODO 0723 注意力
        self.atten_3 = Conv(2, 1, 3)
        self.sigmoid = nn.Sigmoid()
        ################################
        ## TODO 综合

        # self.asff = ASFF(c2 // 2, c2 - (c2 // 2), self.init_channels + 1)  ## TODO 0713
        self.endconv = Conv(c2 + self.init_channels + 1, c2, k, s, p)

    def forward(self, x):
        ## TODO 选取重要通道 0712  (7.23能不能将特征表示强的通道使用Ghost模块生成剩余的通道，在相加)
        scales = self.eca(x).view(x.size(0), -1)
        _, positive_channels = torch.sort(scales, dim=1, descending=True)  ## 从大到小排序，返回是最大值对应的索引
        positive_channel = positive_channels[:, :self.init_channels].split(1)
        channel_tmp = torch.zeros(x.size(0), x.size(1)).bool().split(1)
        x_posi = torch.ones(x.size(0), self.init_channels, x.size(2), x.size(3)).type_as(x)
        if self.init_channels != self.ch_1:
            x_nega = torch.ones(x.size(0), x.size(1) - self.init_channels, x.size(2), x.size(3)).type_as(x)
            for i in range(x.size(0)):
                channel_tmp[i][:, positive_channel[i].view(-1)] = True
                x_posi[i] = x[i][channel_tmp[i].view(-1), :, :]
                x_nega[i] = x[i][~channel_tmp[i].view(-1), :, :]
            out_x = torch.cat((x_posi, torch.mean(x_nega, dim=1).unsqueeze(dim=1)), dim=1)
        else:
            for i in range(x.size(0)):
                channel_tmp[i][:, positive_channel[i].view(-1)] = True
                x_posi[i] = x[i][channel_tmp[i].view(-1), :, :]
                out_x = x_posi
        ## TODO 通道特征分别计算 0711
        split1, split2 = torch.split(x, x.size(1) // self.r, dim=1)
        tmp_out_1 = self.conv_2(split1)

        _tmp = self.encode1(split2)
        _tmp = self.decode1(_tmp)
        _tmp = self.encode2(_tmp)
        tmp_out_2 = self.decode2(_tmp)

        ## TODO 7.23
        commpress_1 = torch.cat(
            (torch.mean(x, dim=1).unsqueeze(dim=1), torch.max(x, dim=1)[0].unsqueeze(dim=1)), dim=1)
        atten_1 = self.sigmoid(self.atten_1(commpress_1)) + 1
        compress_2 = torch.cat(
            (torch.mean(tmp_out_1, dim=1).unsqueeze(dim=1), torch.max(tmp_out_1, dim=1)[0].unsqueeze(dim=1)), dim=1)
        atten_2 = self.sigmoid(self.atten_1(compress_2 * atten_1)) + 1
        compress_3 = torch.cat(
            (torch.mean(tmp_out_2, dim=1).unsqueeze(dim=1), torch.max(tmp_out_2, dim=1)[0].unsqueeze(dim=1)), dim=1)
        atten_3 = self.sigmoid(self.atten_1(compress_3 * atten_1)) + 1
        #####################################################################
        outs = torch.cat((tmp_out_1 * atten_2, tmp_out_2 * atten_3, out_x), dim=1)  ##TODO 0712 将所需要的内容残差过来
        outs = self.endconv(outs)
        return outs


def count(lists):
    for list in lists:
        _, c, _, _ = list.size()
        scores_1 = F.adaptive_avg_pool2d(list, 1)
        scores_2 = F.adaptive_max_pool2d(list, 1)
        # score_1 = F.softmax(scores_1,dim=1)
        score_1 = F.sigmoid(scores_1)

        y_1 = score_1.view(1, -1).squeeze().detach().numpy()
        sort_y = np.sort(y_1)
        means = np.mean(y_1).repeat(y_1.shape)

        y_2 = scores_2.view(1, -1).squeeze().detach().numpy()
        x = np.arange(1, c + 1)
        plt.plot(x, y_1, 's-', color='r', label="adaptive_avg_pool2d")  # s-:方形
        plt.plot(x, means, 'o-', color='g', label="adaptive_max_pool2d")  # s-:方形
        plt.plot(x, sort_y, 'o-', color='y', label="adaptive_max_pool2d")  # s-:方形
        plt.text(1, means[0], means[0], ha='center', va='bottom', fontsize=20)

        plt.xlabel("score")  # 横坐标名字
        plt.ylabel("channels")  # 纵坐标名字
        plt.legend(loc="best")  # 图例
        # plt.savefig("count.png")
        plt.show()


if __name__ == '__main__':
    from torchstat import stat

    inputs = torch.rand(1, 64, 124, 120) * -1
    model = SAS(64, 256, 3, 2, g=4)
    # model = ContextDetail(32, 64, 3, 2)
    # model = MutilSpatial(32,64)
    # model = SplAtConv2d(32, 64, 3, groups=2)
    # model = Conv(32, 64, 3)
    # model = ResShrink(32, 32)
    # model = ShrinkAttention(32)
    # stat(model, (32, 320, 320))
    out = model(inputs)
    # model = MutilSpatial(1024, 32).cuda()
    # model = SplAtConv2d(32, 64, 1, groups=2)
    # outs = model(inputs)
    print()

"""
SAS
Total params: 2,016
-----------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 70.31MB
Total MAdd: 128.2MMAdd
Total Flops: 71.07MFlops
Total MemR+W: 104.7MB

Conv
Total params: 18,560
---------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 75.00MB
Total MAdd: 3.79GMAdd
Total Flops: 1.91GFlops
Total MemR+W: 137.57MB
"""
