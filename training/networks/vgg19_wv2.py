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
import numpy as np
import cv2
import pywt

logger = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="vgg19_wv2")
class Vgg19_wv2(nn.Module):
    def __init__(self, vgg_config):
        super(Vgg19_wv2, self).__init__()
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
        # self.attention_x = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
        #     nn.PReLU(),
        #     torch.nn.BatchNorm2d(512),
        #     #torch.nn.Sigmoid()
        # )
        # self.attention_y = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
        #     nn.PReLU(),
        #     torch.nn.BatchNorm2d(512),
        #     #torch.nn.Sigmoid()
        # )
        # self.attention_xy = torch.nn.Sequential(
        #     torch.nn.Conv2d(512, 512, kernel_size=1, stride=1),
        #     #nn.PReLU(),
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.Sigmoid()
        # )
        # # spatial attention
        # self.spatial_attention = torch.nn.Sequential(
        #     torch.nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1),
        #     #torch.nn.PReLU(),
        #     torch.nn.BatchNorm2d(1),
        #     torch.nn.Sigmoid()
        # )

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
        
        # self.vgg = vgg.features
        # self.avgpool = vgg.avgpool
        # self.classify = vgg.classifier

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(512, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )

    # def resize_coefficients(self,coefficients_batch):
    #     resized_coeffs = []
    #     for coefficients in coefficients_batch:
    #         # resized_coeffs = []
    #         counter = 0
    #         for channel_coeffs in coefficients:
    #             if counter != 0:
    #                 resized_channel_coeffs = [cv2.resize(coeff, (256, 256))[:,:,np.newaxis] for coeff in channel_coeffs]
    #                 resized_channel_coeffs = np.concatenate(resized_channel_coeffs, axis = 2)
    #                 resized_coeffs.append(resized_channel_coeffs)
    #             counter = counter + 1
    #         # resized_coeffs = [cv2.resize(coeff, (224, 224)) for coeff in coefficients]
    #         # combined_coeffs = [np.concatenate(channel_coeffs, axis = 2) for channel_coeffs in resized_coeffs]
    #     # return np.concatenate(resized_coeffs, axis = 2)
    #     return resized_coeffs

    def apply_wavelet_transform(self,channel):
        coeff_batch =[]
        for im in channel:
            coeffs = pywt.wavedec2(im, 'db2', level=1)  # Adjust the wavelet type and level as needed
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
        image_rec = [pywt.waverec2(coeffs, 'db2') for coeffs in wavelet_coeffs]
        
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
        #tensor_image = self.transform(img)
        
        x = self.features_1(wavelet_tensor.cuda())

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
        # x_conv = self.attention_x(x)
        # #w_xy = self.soft_attention(x)
        # y_conv = self.attention_y(x)
        # temp_conv = self.attention_xy(x_conv + y_conv)
        # x = x_conv + (x_conv * temp_conv)
        # b, c, h, w = x.size()
        # scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # scale = self.spatial_attention(scale)
        # #temp = torch.nn.functional.softmax(scale,dim=1)
        # #temp1 = torch.nn.functional.softmax(scale,dim=0)
        # #scale = temp + temp1
        # scale = torch.exp(scale)
        # #print(scale.shape)
        # temp = torch.sum(scale,(2,3))
        # #print(temp.shape)
        # scale = scale / temp.unsqueeze(1).unsqueeze(1)
        # x = x + (x * scale)
        
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
