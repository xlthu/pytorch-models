"""
Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import operator
from functools import reduce

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_blocks, in_c, channels, stride1, is_basic=True):
        super(Block, self).__init__()
        self.n_blocks = n_blocks

        self.in_c = in_c
        self.channels = channels
        self.out_c = self.channels * (1 if is_basic else 4)
        self.stride1 = stride1
        
        self.downsample = self._makeDownsample()
        self._makeBlock = self._makeBasicBlock if is_basic else self._makeBottleneckBlock

        self.blocks = [self._makeBlock(in_c=self.in_c, stride=stride1)]
        for i in range(1, n_blocks):
            self.blocks.append(self._makeBlock(in_c=self.out_c, stride=1))

        self.blocks = nn.ModuleList(self.blocks)
        self.relu = nn.ReLU(inplace=True)

    def _makeDownsample(self):
        if self.stride1 != 1 or self.in_c != self.out_c:
            return nn.Sequential(
                nn.Conv2d(self.in_c, self.out_c, kernel_size=1, stride=self.stride1, bias=False),
                nn.BatchNorm2d(self.out_c)
            )
        else:
            return None

    def _makeBasicBlock(self, in_c, stride=1):
        channels = self.channels
        return nn.Sequential(
                nn.Conv2d(in_c, channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )

    def _makeBottleneckBlock(self, in_c, stride=1):
        channels = self.channels
        out_c = self.out_c
        return nn.Sequential(
                nn.Conv2d(in_c, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(channels, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
    
    def forward(self, x):
        down_x = self.downsample(x) if self.downsample else x
        out = self.blocks[0](x)
        x = self.relu(out + down_x)

        for i in range(1, self.n_blocks):
            out = self.blocks[i](x)
            x = self.relu(out + x)
        
        return x

class Resnet(nn.Module):
    def __init__(self, input_size, num_classes, cfg):
        super(Resnet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes        
        self.features, out_size = self._makeFeatures(cfg)
        self.classifier = self._makeClassifier(out_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _makeFeatures(self, cfg):
        is_basic = cfg['is_basic']
        blocks = cfg['blocks']
        channels = cfg['channels']

        assert(len(blocks) >= 1 and len(blocks) == len(channels))
    
        out_frac = 2 ** (len(blocks) + 1)
        assert(self.input_size[1] % out_frac == 0 or self.input_size[2] % out_frac == 0)
        out_feature_size = (self.input_size[1] // out_frac, self.input_size[2] // out_frac)
        
        features = [
            nn.Conv2d(self.input_size[0], 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        ]

        in_c = 64
        for i, (n, c) in enumerate(zip(blocks, channels)):
            stride1 = 1 if i == 0 else 2
            block = Block(n, in_c, c, stride1=stride1, is_basic=is_basic)
            features.append(block)
            in_c = block.out_c

        features.append(nn.AvgPool2d(kernel_size=out_feature_size))

        return nn.Sequential(*features), in_c

    def _makeClassifier(self, in_size):
        return nn.Linear(in_size, self.num_classes)

    def initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def Resnet18(input_size, num_classes):
    cfg = {
        'is_basic': True,
        'blocks': [2, 2, 2, 2],
        'channels': [64, 128, 256, 512]
    }
    return Resnet(input_size, num_classes, cfg=cfg)

def Resnet34(input_size, num_classes):
    cfg = {
        'is_basic': True,
        'blocks': [3, 4, 6, 3],
        'channels': [64, 128, 256, 512]
    }
    return Resnet(input_size, num_classes, cfg=cfg)

def Resnet50(input_size, num_classes):
    cfg = {
        'is_basic': False,
        'blocks': [3, 4, 6, 3],
        'channels': [64, 128, 256, 512]
    }
    return Resnet(input_size, num_classes, cfg=cfg)

def Resnet101(input_size, num_classes):
    cfg = {
        'is_basic': False,
        'blocks': [3, 4, 23, 3],
        'channels': [64, 128, 256, 512]
    }
    return Resnet(input_size, num_classes, cfg=cfg)

def Resnet152(input_size, num_classes):
    cfg = {
        'is_basic': False,
        'blocks': [3, 8, 36, 3],
        'channels': [64, 128, 256, 512]
    }
    return Resnet(input_size, num_classes, cfg=cfg)