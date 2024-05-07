'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for Vgg19 backbone.
'''

import os
import logging
from typing import Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import BACKBONE

logger = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="vgg19")
class Vgg19(nn.Module):
    def __init__(self, vgg_config):
        super(Vgg19, self).__init__()
        """ Constructor
        Args:
            vgg_config: configuration file with the dict format
        """
        self.num_classes = vgg_config["num_classes"]
        inc = vgg_config["inc"]
        self.mode = vgg_config["mode"]

        # Define layers of the backbone
        vgg = torchvision.models.vgg19_bn(pretrained=True)  # FIXME: download the pretrained weights from online
        vgg.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
        # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vgg = vgg.features
        self.avgpool = vgg.avgpool
        self.classify = vgg.classifier

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(512, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )


    def features(self, inp):
        x = self.vgg(inp)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out
