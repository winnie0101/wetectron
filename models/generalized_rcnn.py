# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from .backbone.vgg16 import VGG_Base
from collections import OrderedDict
from models.rpn.retinanet.retinanet import RetinaNetModule
from models.rpn.rpn import RPNModule  
from models.roi_heads.weak_head.weak_head import ROIWeakRegHead  
from .ccam_head import CCAMHead


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.body = VGG_Base(cfg)
        model = nn.Sequential(OrderedDict([("body", self.body)]))
        model.out_channels = 512
        self.backbone = model
        self.cfg = cfg

        if cfg.MODEL.FASTER_RCNN:
            if cfg.MODEL.RETINANET_ON:
                self.rpn = RetinaNetModule(cfg, self.backbone.out_channels)
            else:
                self.rpn = RPNModule(cfg, self.backbone.out_channels)

        self.roi_heads = ROIWeakRegHead(cfg, self.backbone.out_channels)
        if cfg.MODEL.CAM_ON:
            self.ccam_head = CCAMHead(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, rois=None, model_cdb=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")       
        features = self.backbone(images.tensors)
        if self.cfg.MODEL.CAM_ON:
            self.feature_map = self.body.get_cam()

        if not self.training and self.cfg.MODEL.CAM_ON:
            self.feature_map = self.body.get_cam()
            
        if rois is not None and rois[0] is not None:
            # use pre-computed proposals
            proposals = rois
            proposal_losses = {}
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses, accuracy = self.roi_heads(features, proposals, targets, model_cdb)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # 分兩路train
        #TODO: 算nl ccam的loss?
        if self.cfg.MODEL.CAM_ON:
            nl_ccam_output = self.ccam_head(self.feature_map)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, accuracy
        elif not self.training and self.cfg.MODEL.CAM_ON:
            return result, self.feature_map
        
        return result

    def backbone_forward(self, images):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed

        Returns:
            features (list[Tensor]): the output from the backbone.
        """    
        return self.backbone(images.tensors)

    def neck_head_forward(self, features, targets=None, rois=None, model_cdb=None):
        """
        Arguments:
            features (list[Tensor]): the output from the backbone.
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            the same as `forward`
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")       

        # use pre-computed proposals
        assert rois is not None
        assert rois[0] is not None
        x, result, detector_losses, accuracy = self.roi_heads(features, rois, targets, model_cdb)

        if self.training:
            return detector_losses, accuracy

        return result

    
    def get_cam(self):
        return self.feature_map