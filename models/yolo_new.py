import argparse
import logging
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common_new import *
from models.experimental import *
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)

class Detect(nn.Module):  # 定义检测网络
    '''
    input:(number_classes, anchors=(), ch=(tensor_small,tensor_medium,tensor_large))    tensor[i]:(batch_size, in_channels, size1, size2)
    size1[i] = img_size1/(8*i)  size2[i] = img_size2/(8*i)   eg:  tensor_small:(batch_size, inchannels, img_size1/8. img_size2/8)
    '''
    stride = None  # strides computed during build
    export = False  # onnx export,网络模型输出为onnx格式，可在其他深度学习框架上运行

    def __init__(self, nc=1, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.angle = 180  # CSL---180  KLD--1
        self.no = nc + 5 + self.angle   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid   [tensor([0.]), tensor([0.]), tensor([0.])] 初始化网格
        # anchor.shape= (3 , 6) -> shape= ( 3 , 3 , 2)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # shape(3, ?(3), 2)
        # register_buffer用法：内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        self.register_buffer('anchors', a)  # shape(nl,na,2) = (3, 3, 2)
        # shape(3, 3, 2) -> shape(3, 1, 3, 1, 1, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,？(na),1,1,2) = (3, 1, 3, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        #if profile:
        #x = x.copy()  # for profiling --must do it

        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    #self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):  # 绘制网格
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):  # 有预训练权重文件时cfg加载权重中保存的cfg字典内容；
            self.yaml = cfg  # model dict
        else:  # is *.yaml 没有预训练权重文件时加载用户定义的opt.cfg权重文件路径，再载入文件中的内容到字典中
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        # Define model
        if nc and nc != self.yaml['nc']:  # 字典中的nc与data.yaml中的nc不同，则以data.yaml中的nc为准
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            x = m(x)
            y.append(x if m.i in self.save else None)  # save output
        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):  # 如果函数层名为Conv标准卷积层，且同时 层中包含‘bn’属性名
                #m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm 将'bn'属性删除
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def add_nms(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if type(self.model[-1]) is not NMS:  # if missing NMS
            print('Adding NMS module... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))  # 打印相关参数的类名
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors  6//2=3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) = 3*85 =255
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out  []  []  3
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except :
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain,BottleneckCSP层中Bottleneck层的个数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3STR, C3SPP, C3Ghost, ASPP, CBAM, nn.ConvTranspose2d]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]  # [ch[-1], out_channels, kernel_size, strides(可能)] — 除了BottleneckCSP与C3层
            if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost]:
                args.insert(2, n)       # [ch[-1], out_channnels, Bottleneck_num] — BottleneckCSP与C3层
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        # 将'__main__.Detect'变为Detect，其余模块名不变，相当于所有函数名全都放在了t中
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 返回当前module结构中参数的总数目
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # 对应相关参数的类名，打印对应参数
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        # 把Concat，Detect需要使用到的参数层的层数信息储存进save
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将每层结构的函数名拓展进layers list
        layers.append(m_)
        # 将每层结构的out_channels拓展进ch，以便下一层结构调用上一层的输出通道数 yolov5.yaml中的第0层的输出对应ch[1] ;i - ch[i+1]
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)

if __name__ == '__main__':
    # import os
    # a = os.path.join("%s", "JPEGImages-val", "%s.png")
    # print(a)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='../weights/yolov5s.pt', help='initil weights path')
    parser.add_argument('--cfg', type=str, default='yolov5s-cbam.yaml', help='model.yaml')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    #Profile
    img = torch.rand(1 if torch.cuda.is_available() else 1, 3, 1024, 1024).to(device)
    print('img:',img.shape)
    y = model(img, profile=True)
    for item in y:
        print(item.shape)
    print('----------over-----------')
