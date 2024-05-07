'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
'''

import os
import argparse
import logging

import math
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pywt

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from utils.registry import BACKBONE

logger = logging.getLogger(__name__)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


@BACKBONE.register_module(module_name="xception_wv2")
class Xception_wv2(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception_wv2, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Linear(2048, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
           
    def apply_wavelet_transform(self,channel):
        coeff_batch =[]
        for im in channel:
            coeffs = pywt.wavedec2(im, 'haar', level=1)  # Adjust the wavelet type and level as needed
            coeff_batch.append(coeffs)
        return coeff_batch
    
    
    def threshold_coefficients(self,coefficients):
        # resized_coeffs = []
        for coeffs in coefficients:
            counter = 0
            for channel_coeffs in coeffs:
                if counter != 0:
                    # resized_channel_coeffs = [cv2.resize(coeff, (224, 224), interpolation = cv2.INTER_AREA)[:,:,np.newaxis] for coeff in channel_coeffs]
                    # resized_channel_coeffs = np.concatenate(resized_channel_coeffs, axis = 2)
                    # resized_coeffs.append(resized_channel_coeffs)
                    channel_coeffs[0][channel_coeffs[0] <= (np.mean(channel_coeffs[0]) - np.std(channel_coeffs[0]))] = 0
                    channel_coeffs[1][channel_coeffs[1] <= (np.mean(channel_coeffs[1]) - np.std(channel_coeffs[1]))] = 0
                    channel_coeffs[2][channel_coeffs[2] <= (np.mean(channel_coeffs[2]) - np.std(channel_coeffs[2]))] = 0
                    # coeffs[:,:,channel_coeffs][coeffs[:,:,channel_coeffs] <= (np.mean(coeffs[:,:,channel_coeffs]) - np.std(coeffs[:,:,channel_coeffs]))] = 0
                else:
                    channel_coeffs = 0 * channel_coeffs
                counter = counter + 1
            # resized_coeffs = [cv2.resize(coeff, (224, 224)) for coeff in coefficients]
            # combined_coeffs = [np.concatenate(channel_coeffs, axis = 2) for channel_coeffs in resized_coeffs]
        return coefficients
    
    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  

        return x

    def fea_part1_1(self, x):  
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)     
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x
     
    def features(self, inp):
        tensor_inp = torch.permute(inp, (2,3,1,0))
        
        # Step 2: Apply the Wavelet Transform
        # tensor_inp = torch.permute(tensor_inp, (1, 2, 0))
        # image = tensor_image.numpy()
        inp = tensor_inp.cpu().numpy()
        
        image = [cv2.cvtColor(inp[:,:,:,im], cv2.COLOR_RGB2GRAY) for im in range(inp.shape[3])]
        
        wavelet_coeffs = self.apply_wavelet_transform(image)
        # print(image.shape)
        
        wavelet_coeffs = self.threshold_coefficients(wavelet_coeffs)
        image_rec = [pywt.waverec2(coeffs, 'haar') for coeffs in wavelet_coeffs]
        
        # Resize coefficients for each channel to 256x256
        # wavelet_resized_coeffs = self.resize_coefficients(wavelet_coeffs)
        
        # combined_coeffs = [np.concatenate([inp[:,:,:,im], wavelet_resized_coeffs[im]], axis = 2)[:,:,:,np.newaxis] for im in range(inp.shape[3])]
        
        combined_coeffs = [np.concatenate([inp[:,:,:,im], image_rec[im][:,:,np.newaxis]], axis = 2)[:,:,:,np.newaxis] for im in range(inp.shape[3])]
        
        # Combine resized coefficients into a single tensor
        combined_coeffs = np.concatenate(combined_coeffs, axis = 3)
        
        # Normalize and add a batch dimension
        # max_coeff_value = np.max(combined_coeffs)  # Adjust this based on your coefficients
        # normalized_coeffs = combined_coeffs / max_coeff_value
        wavelet_tensor = torch.tensor(combined_coeffs, dtype=torch.float32)
        # Step 3: Convert the Wavelet Coefficients to a Tensor
        # wavelet_tensor = torch.tensor(wavelet_coeffs, dtype=torch.float32)
        wavelet_tensor = torch.permute(wavelet_tensor, (3,2,0,1))
        x = self.fea_part1(wavelet_tensor.cuda())    

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        
        return x

    def classifier(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x
