# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch

from ..box_head.inference import make_roi_box_post_processor as strong_roi_box_post_processor

from .inference import make_roi_box_post_processor as weak_roi_box_post_processor
from .roi_sampler import make_roi_sampler

from utils.utils import cat
from utils.boxlist_ops import cat_boxlist
from models.backbone.vgg16 import VGG16FC67ROIFeatureExtractor
from .roi_weak_predictors import MISTPredictor
from .loss import RoIRegLossComputation


class ROIWeakRegHead(torch.nn.Module):
    """ Generic Box Head class w/ regression. """
    def __init__(self, cfg, in_channels):
        super(ROIWeakRegHead, self).__init__()

        self.feature_extractor = VGG16FC67ROIFeatureExtractor(cfg, in_channels)
        self.predictor = MISTPredictor(cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = RoIRegLossComputation(cfg)
        
        self.weak_post_processor = weak_roi_box_post_processor(cfg)
        self.strong_post_processor = strong_roi_box_post_processor(cfg)
        
        self.HEUR = cfg.MODEL.ROI_WEAK_HEAD.REGRESS_HEUR
        self.roi_sampler = make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None
        self.DB_METHOD = cfg.DB.METHOD

    def go_through_cdb(self, features, proposals, model_cdb):
        if not self.training or self.DB_METHOD == "none":
            return self.feature_extractor(features, proposals)
        elif self.DB_METHOD == "concrete":
            x = self.feature_extractor.forward_pooler(features, proposals)
            x = model_cdb(x)
            return self.feature_extractor.forward_neck(x)
        else:
            raise ValueError
        
    def forward(self, features, proposals, targets=None, model_cdb=None):
        # for partial labels
        if self.roi_sampler is not None and self.training:
            with torch.no_grad():
                proposals = self.roi_sampler(proposals, targets)
        roi_feats  = self.go_through_cdb(features, proposals, model_cdb)
        cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(roi_feats, proposals)
        if not self.training:
            result = self.testing_forward(cls_score, det_score, proposals, ref_scores, ref_bbox_preds)
            return roi_feats, result, {}, {}
        loss_img, accuracy_img = self.loss_evaluator([cls_score], [det_score], ref_scores, ref_bbox_preds, proposals, targets)
        return (roi_feats, proposals, loss_img, accuracy_img)

    def testing_forward(self, cls_score, det_score, proposals, ref_scores=None, ref_bbox_preds=None):
        if self.HEUR == "WSDDN":
            final_score = cls_score * det_score
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "CLS-AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "AVG": # AVG
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            final_regression = torch.mean(torch.stack(ref_bbox_preds), dim=0)
            result = self.strong_post_processor((final_score, final_regression), proposals, softmax_on=False)
        elif self.HEUR == "UNION": # UNION
            prop_list = [len(p) for p in proposals]
            ref_score_list = [rs.split(prop_list) for rs in ref_scores]
            ref_bbox_list = [rb.split(prop_list) for rb in ref_bbox_preds]
            final_score = [torch.cat((ref_score_list[0][i], ref_score_list[1][i], ref_score_list[2][i])) for i in range(len(proposals)) ]
            final_regression = [torch.cat((ref_bbox_list[0][i], ref_bbox_list[1][i], ref_bbox_list[2][i])) for i in range(len(proposals)) ]
            augmented_proposals = [cat_boxlist([p for _ in range(3)]) for p in proposals]
            result = self.strong_post_processor((cat(final_score), cat(final_regression)), augmented_proposals, softmax_on=False)
        else:
            raise ValueError
        return result

