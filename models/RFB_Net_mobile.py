import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=stride, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1,x2),1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        out = out*self.scale + x
        out = self.relu(out)

        return out

class RFBNet(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 1
        else:
            print("Error: Sorry only RFB300_mobile is supported!")
            return

        self.base = nn.ModuleList(base)
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(12):
            x = self.base[k](x)


        s = self.Norm(x)
        sources.append(s)

        for k in range(12, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 == 0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def conv_bn(inp,oup,stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
            nn.Conv2d(inp,inp, kernel_size=3, stride=stride, padding=1,groups = inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
    )

def MobileNet():
    layers = []
    layers += [conv_bn(3, 32, 2)]
    layers += [conv_dw(32, 64, 1)]
    layers += [conv_dw(64, 128, 2)]
    layers += [conv_dw(128, 128, 1)]
    layers += [conv_dw(128, 256, 2)]
    layers += [conv_dw(256, 256, 1)]
    layers += [conv_dw(256, 512, 2)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 1024, 2)]
    layers += [conv_dw(1024, 1024, 1)]

    return layers



def add_extras(size, cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0)]
        in_channels = v
    if size ==300:
        layers += [BasicConv(512,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=2, padding=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=2, padding=1)]
        layers += [BasicConv(256,64,kernel_size=1,stride=1)]
        layers += [BasicConv(64,128,kernel_size=3,stride=2, padding=1)]
    else:
        print("Error: Sorry only RFB300_mobile is supported!")
        return
    return layers

extras = {
    '300': ['S', 512 ],
}


def multibox(size, base, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    base_net= [-2,-1]
    for k, v in enumerate(base_net):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=1, padding=0)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=1, padding=0)]
        else:
            loc_layers += [nn.Conv2d(1024,
                                 cfg[k] * 4, kernel_size=1, padding=0)]
            conf_layers += [nn.Conv2d(1024,
                        cfg[k] * num_classes, kernel_size=1, padding=0)]
    i = 2
    indicator = 0
    if size == 300:
        indicator = 1
    else:
        print("Error: Sorry only RFB300_mobile is supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=1, padding=0)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=1, padding=0)]
            i +=1
    return base, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only RFB300_mobile is supported!")
        return

    return RFBNet(phase, size, *multibox(size, MobileNet(),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
