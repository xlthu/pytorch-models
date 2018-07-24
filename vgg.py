import operator
from functools import reduce

import torch
import torch.nn as nn

class VGG(nn.Module):
    """VGG Model"""
    def __init__(self, input_size, num_classes, cfg):
        super(VGG, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Model
        self.features, out_size = self._makeFeatures(cfg)
        self.classifier = self._makeClassifier(out_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _makeFeatures(self, cfg):
        def conv3(in_channels, out_channels):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    ]

        def maxpool():
            return nn.MaxPool2d(kernel_size=2, stride=2)

        layers = []
        in_channels = self.input_size[0]
        out_frac = 1
        for out_channels in cfg:
            if out_channels == 'M':
                layers.append(maxpool())
                out_frac *= 2
            else:
                layers.extend(conv3(in_channels, out_channels))
                in_channels = out_channels

        assert(self.input_size[1] % out_frac == 0 or self.input_size[2] % out_frac == 0)

        out_shape = (in_channels, self.input_size[1] // out_frac, self.input_size[2] // out_frac)
        return nn.Sequential(*layers), out_shape

    def _makeClassifier(self, in_shape):
        return nn.Sequential(
            nn.Linear(in_features=reduce(operator.mul, in_shape, 1), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes),
            )

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

def VGG16(input_size, num_classes):
    """Configuration D, VGG16"""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(input_size, num_classes, cfg=cfg)
