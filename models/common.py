# This file contains modules common to various models
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import non_max_suppression
'''
feature map尺寸计算公式： out_size = (in_size + 2*Padding - kernel_size)/strides + 1
卷积计算时map尺寸向下取整
池化计算时map尺寸向上取整
'''

def autopad(k, p=None):  # kernel, padding
    '''
    自动填充
    返回padding值
        kernel_size 为 int类型时 ：padding = k // 2（整数除法进行一次）
                        否则    : padding = [x // 2 for x in k]
    '''
    # Pad to 'same'
    if p is None:  # k是否为int类型，是则返回True
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def DWConv(c1, c2, k=1, s=1, act=True):
    '''
    深度分离卷积层 Depthwise convolution：
        是G（group）CONV的极端情况；
        分组数量等于输入通道数量，即每个通道作为一个小组分别进行卷积，结果联结作为输出，Cin = Cout = g，没有bias项。
        c1 : in_channels
        c2 : out_channels
        k : kernel_size
        s : stride
        act : 是否使用激活函数
        math.gcd() 返回的是最大公约数
    '''
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    '''
    标准卷积层Conv
    包括Conv2d + BN + HardWish激活函数
    (self, in_channels, out_channels, kernel_size, stride, padding, groups, activation_flag)
    p=None时，out_size = in_size/strides
    '''
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):  # 前向计算（有BN）
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):  # 前向融合计算（无BN）
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    '''
    标准BasicBlock层
        input : input
        output : input + Conv3×3（Conv3×3(input)）
    (self, in_channels, out_channels, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''
    def __init__(self, c1, c2, s=1,g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(BasicBlock, self).__init__()
        self.cv1 = Conv(c1, c2, 3, s,g=g)
        self.cv2 = Conv(c2, c2, 3, 1,g=g)
       # self.add = shortcut and c1 == c2

    def forward(self, x):
        '''
        若 shortcut_flag为Ture 且 输入输出通道数相等，则返回跳接后的结构：
            x + Conv3×3（Conv1×1(x)）
        否则不进行跳接：
            Conv3×3（Conv1×1(x)）
        '''
        return x + self.cv2(self.cv1(x)) #if self.add else self.cv2(self.cv1(x))

class Bottleneck(nn.Module):
    '''
    标准Bottleneck层
        input : input
        output : input + Conv3×3（Conv1×1(input)）
    (self, in_channels, out_channels, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        '''
        若 shortcut_flag为Ture 且 输入输出通道数相等，则返回跳接后的结构：
            x + Conv3×3（Conv1×1(x)）
        否则不进行跳接：
            Conv3×3（Conv1×1(x)）
        '''
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck_Cbam(nn.Module):
    '''
    标准Bottleneck层
        input : input
        output : input + Conv3×3（Conv1×1(input)）
    (self, in_channels, out_channels, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_Cbam, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        #CBMA
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        '''
        若 shortcut_flag为Ture 且 输入输出通道数相等，则返回跳接后的结构：
            x + Conv3×3（Conv1×1(x)）
        否则不进行跳接：
            Conv3×3（Conv1×1(x)）
        '''
        residual = x
        out = self.cv2(self.cv1(x))
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.add:
            out += residual
        return out
        #return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP_Cbam(nn.Module):
    '''
    标准ottleneckCSP层
    (self, in_channels, out_channels, Bottleneck层重复次数, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP_Cbam, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck_Cbam(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))   # CONV + BottleNeck + Conv2d  out_channels = c_
        y2 = self.cv2(x)  # Conv2d   out_channels = c_
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # concat(y1 + y2) + BN + LeakyReLU + Conv2d  out_channels = c2


class BottleneckCSP(nn.Module):
    '''
    标准ottleneckCSP层
    (self, in_channels, out_channels, Bottleneck层重复次数, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''
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
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))   # CONV + BottleNeck + Conv2d  out_channels = c_
        y2 = self.cv2(x)  # Conv2d   out_channels = c_
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # concat(y1 + y2) + BN + LeakyReLU + Conv2d  out_channels = c2

class SPP(nn.Module):
    '''
    空间金字塔池化SPP：
    (self, in_channels, out_channels, 池化尺寸strides[3])
    '''
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 建立5×5 9×9 13×13的最大池化处理过程的list
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Focus(nn.Module):
    '''
    Focus : 把宽度w和高度h的信息整合到c空间中
    (self, in_channels, out_channels, kernel_size, stride, padding, group, activation_flag)
    '''
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        '''
        x(batch_size, channels, height, width) -> y(batch_size, 4*channels, height/2, weight/2)
        '''
        # ::代表[start:end:step], 以2为步长取值
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    '''
    (dimension)
    默认d=1按列拼接 ， d=0则按行拼接
    '''
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

class Flatten(nn.Module):
    '''
    在全局平均池化以后使用，去掉2个维度
    (batch_size, channels, size, size) -> (batch_size, channels*size*size)
    '''
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)

class Classify(nn.Module):
    '''
    (self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1)
    (batch_size, channels, size, size) -> (batch_size, channels*1*1)
    '''
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        # 给定输入数据和输出数据的大小，自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(batch_size,ch_in,1,1) 返回1×1的池化结果
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(batch_size,ch_out,1,1)
        self.flat = Flatten()

    def forward(self, x):
        #
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if x is list
        return self.flat(self.conv(z))  # flatten to x(batch_size, ch_out×1×1)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class CBAM(nn.Module):
#
#     def __init__(self, n_channels_in, reduction_ratio, kernel_size):
#         super(CBAM, self).__init__()
#         self.n_channels_in = n_channels_in
#         self.reduction_ratio = reduction_ratio
#         self.kernel_size = kernel_size
#
#         self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
#
#     def forward(self, f):
#         chan_att = self.channel_attention(f)
#         # print(chan_att.size())
#         fp = chan_att * f
#         # print(fp.size())
#         spat_att = self.spatial_attention(fp)
#         # print(spat_att.size())
#         fpp = spat_att * fp
#         # print(fpp.size())
#         return fpp
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size):
#         super(SpatialAttention, self).__init__()
#         self.kernel_size = kernel_size
#
#         assert kernel_size % 2 == 1, "Odd kernel size required"
#         self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
#                               padding=int((kernel_size - 1) / 2))
#         # batchnorm
#
#     def forward(self, x):
#         max_pool = self.agg_channel(x, "max")
#         avg_pool = self.agg_channel(x, "avg")
#         pool = torch.cat([max_pool, avg_pool], dim=1)
#         conv = self.conv(pool)
#         # batchnorm ????????????????????????????????????????????
#         conv = conv.repeat(1, x.size()[1], 1, 1)
#         att = torch.sigmoid(conv)
#         return att
#
#     def agg_channel(self, x, pool="max"):
#         b, c, h, w = x.size()
#         x = x.view(b, c, h * w)
#         x = x.permute(0, 2, 1)
#         if pool == "max":
#             x = F.max_pool1d(x, c)
#         elif pool == "avg":
#             x = F.avg_pool1d(x, c)
#         x = x.permute(0, 2, 1)
#         x = x.view(b, 1, h, w)
#         return x
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, n_channels_in, reduction_ratio):
#         super(ChannelAttention, self).__init__()
#         self.n_channels_in = n_channels_in
#         self.reduction_ratio = reduction_ratio
#         self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))
#
#         self.bottleneck = nn.Sequential(
#             nn.Linear(self.n_channels_in, self.middle_layer_size),
#             nn.ReLU(),
#             nn.Linear(self.middle_layer_size, self.n_channels_in)
#         )
#
#     def forward(self, x):
#         kernel = (x.size()[2], x.size()[3])
#         avg_pool = F.avg_pool2d(x, kernel)
#         max_pool = F.max_pool2d(x, kernel)
#
#         avg_pool = avg_pool.view(avg_pool.size()[0], -1)
#         max_pool = max_pool.view(max_pool.size()[0], -1)
#
#         avg_pool_bck = self.bottleneck(avg_pool)
#         max_pool_bck = self.bottleneck(max_pool)
#
#         pool_sum = avg_pool_bck + max_pool_bck
#
#         sig_pool = torch.sigmoid(pool_sum)
#         sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)
#
#         out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
#         return out
#
#
# def main():
#     # ca = CBAM()
#
#     f = torch.FloatTensor([
#         [
#             [[1, 1, 1, 1, 1], [1, 1, 2, 1, 1], [1, 1, 1, 1, 1]],
#             [[2, 2, 2, 2, 2], [2, 2, 3, 2, 2], [2, 2, 2, 2, 2]],
#             [[3, 3, 3, 3, 3], [3, 3, 4, 3, 3], [3, 3, 3, 3, 3]]
#         ]
#     ])
#
#     print(f.size())
#
#     # sa = SpatialAttention(kernel_size = 3)
#     # sa(f)
#     cbam = CBAM(n_channels_in=f.size()[1], reduction_ratio=2, kernel_size=3)
#
#     fpp = cbam(f)
#     print(fpp.size())
#     print(fpp)
#     # print(f)
#     # print(fp)
#
# if __name__ == "__main__":
#     main()
