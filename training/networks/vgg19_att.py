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

@BACKBONE.register_module(module_name="vgg19_att")
class Vgg19_att(nn.Module):
    def __init__(self, vgg_config):
        super(Vgg19_att, self).__init__()
        """ Constructor
        Args:
            vgg_config: configuration file with the dict format
        """
        self.num_classes = vgg_config["num_classes"]
        inc = vgg_config["inc"]
        self.mode = vgg_config["mode"]

        # Define layers of the backbone
        model = torchvision.models.vgg19_bn(pretrained=True)  # FIXME: download the pretrained weights from online
        # pool layer
        self.pool = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        #self.beta = torch.nn.Parameter(torch.tensor([0.0]))

        # channel attention
        self.attention_x = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.PReLU(),
            torch.nn.BatchNorm2d(512),
            #torch.nn.Sigmoid()
        )
        self.attention_y = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.PReLU(),
            torch.nn.BatchNorm2d(512),
            #torch.nn.Sigmoid()
        )
        self.attention_xy = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
            #nn.PReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Sigmoid()
        )
        # spatial attention
        self.spatial_attention = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1),
            #torch.nn.PReLU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

        # channel attention
        # self.max_pool_1 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=224, stride=224))
        # self.max_pool_2 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=112, stride=112))
        # self.max_pool_3 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=56, stride=56))
        # self.max_pool_4 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=28, stride=28))
        # self.max_pool_5 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=14, stride=14))
        # self.avg_pool_1 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=224, stride=224))
        # self.avg_pool_2 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=112, stride=112))
        # self.avg_pool_3 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=56, stride=56))
        # self.avg_pool_4 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=28, stride=28))
        # self.avg_pool_5 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=14, stride=14))

        # features without considering vgg19 pooling layer
        self.features_1 = torch.nn.Sequential(*list(model.features.children())[:3])
        self.features_2 = torch.nn.Sequential(*list(model.features.children())[3:6])
        self.features_3 = torch.nn.Sequential(*list(model.features.children())[7:10])
        self.features_4 = torch.nn.Sequential(*list(model.features.children())[10:13])
        self.features_5 = torch.nn.Sequential(*list(model.features.children())[14:17])
        self.features_6 = torch.nn.Sequential(*list(model.features.children())[17:20])
        self.features_7 = torch.nn.Sequential(*list(model.features.children())[20:23])
        self.features_8 = torch.nn.Sequential(*list(model.features.children())[23:26])
        self.features_9 = torch.nn.Sequential(*list(model.features.children())[27:30])
        self.features_10 = torch.nn.Sequential(*list(model.features.children())[30:33])
        self.features_11 = torch.nn.Sequential(*list(model.features.children())[33:36])
        self.features_12 = torch.nn.Sequential(*list(model.features.children())[36:39])
        self.features_13 = torch.nn.Sequential(*list(model.features.children())[40:43])
        self.features_14 = torch.nn.Sequential(*list(model.features.children())[43:46])
        self.features_15 = torch.nn.Sequential(*list(model.features.children())[46:49])
        self.features_16 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            torch.nn.ReLU(inplace=True))
        

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        # classifier
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.num_classes)
        )
        
        
        # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes, bias=True)
        # # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.vgg = vgg.features
        # self.avgpool = vgg.avgpool
        # self.classify = vgg.classifier

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(512, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )


    def features(self, inp):
        x = self.features_1(inp)

        x = self.features_2(x)
        
        x = self.pool(x)

        x = self.features_3(x)
        

        x = self.features_4(x)
        
        x = self.pool(x)

        x = self.features_5(x)
        

        x = self.features_6(x)
        

        x = self.features_7(x)
        

        x = self.features_8(x)
        
        x = self.pool(x)

        x = self.features_9(x)
        

        x = self.features_10(x)
        

        x = self.features_11(x)
        

        x = self.features_12(x)
        
        x = self.pool(x)

        x = self.features_13(x)
        

        x = self.features_14(x)
        

        x = self.features_15(x)
        
        
        
        x = self.features_16(x)
        x_conv = self.attention_x(x)
        #w_xy = self.soft_attention(x)
        y_conv = self.attention_y(x)
        temp_conv = self.attention_xy(x_conv + y_conv)
        x = x_conv + (x_conv * temp_conv)
        b, c, h, w = x.size()
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        #temp = torch.nn.functional.softmax(scale,dim=1)
        #temp1 = torch.nn.functional.softmax(scale,dim=0)
        #scale = temp + temp1
        scale = torch.exp(scale)
        #print(scale.shape)
        temp = torch.sum(scale,(2,3))
        #print(temp.shape)
        scale = scale / temp.unsqueeze(1).unsqueeze(1)
        x = x + (x * scale)
        
        x = self.pool(x)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out
