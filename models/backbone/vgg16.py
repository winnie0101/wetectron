# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch.nn as nn
from utils import registry
from utils.poolers import Pooler
from models.non_local import Self_Attn


# to auto-load imagenet pre-trainied weights
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class VGG_Base(nn.Module):
    def __init__(self, cfg, init_weights=True):
        super(VGG_Base, self).__init__()
        if cfg.MODEL.NON_LOCAL:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Self_Attn(128),
    
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Self_Attn(256),
    
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Identity(),
                Self_Attn(512),
                
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                Self_Attn(512),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Identity(),
                
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            )
        if init_weights:
            self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT, cfg.MODEL.NON_LOCAL)

    def forward(self, x):
        x = self.features(x)
        self.feature_map = x
        return [x]
    
    def get_cam(self):
        return self.feature_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self, freeze_at, is_non_local):
        if freeze_at < 0 or not is_non_local:
            return
        
        assert freeze_at in [1, 2, 3, 4, 5]
        layer_index = [5, 10, 17, 23, 29]

        for layer in range(layer_index[freeze_at - 1]):
            for p in self.features[layer].parameters(): p.requires_grad = False


def make_layers(cfg, dim_in=3, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'I':
            layers += [Identity()]
        # following OICR paper, make conv5_x layers to have dilation=2
        elif isinstance(v, str) and '-D' in v:
            _v = int(v.split('-')[0])
            conv2d = nn.Conv2d(in_channels, _v, kernel_size=3, padding=2, dilation=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(_v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = _v          
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # remove the last relu
    return nn.Sequential(*layers[:-1])


vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG16-OICR': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'I', '512-D', '512-D', '512-D'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
    

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("VGG16.roi_head")
class VGG16FC67ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels, init_weights=True):
        super(VGG16FC67ROIFeatureExtractor, self).__init__()
        assert in_channels == 512
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.config = config

        if config.MODEL.FC_TO_CONV:
            self.classifier = nn.Sequential(
                Identity(),
                nn.AdaptiveAvgPool2d((7, 7)),  # 将特征图大小调整为7x7 [4012, 512, 7, 7]
                nn.Conv2d(512, 4096, kernel_size=1),  # 使用1x1卷积进行线性变换 [4012, 4096, 7, 7]
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.AdaptiveAvgPool2d((1, 1)), 
            )
        else:
            self.classifier =  nn.Sequential(
                Identity(),
                nn.Linear(512 * 7 * 7, 4096), #[4012, 4096]
                nn.ReLU(inplace=True), 
                nn.Dropout(), 
                # nn.Linear(4096, 4096),
                # nn.ReLU(inplace=True), 
                # nn.Dropout()
            )
        self.out_channels = 4096
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, proposals):
        # also pool featurs of multiple images into one huge ROI tensor
        x = self.pooler(x, proposals)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward_pooler(self, x, proposals):
        x = self.pooler(x, proposals)
        return x

    def forward_neck(self, x):
        if not self.config.MODEL.FC_TO_CONV:
            x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x