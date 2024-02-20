import torch
from torch import nn

class CCAMHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CCAMHead, self).__init__()
        self.cfg = cfg

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if cfg.MODEL.FC_TO_CONV:
            self.classifier = nn.Conv2d(in_channels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, kernel_size=1)
        else:
            self.classifier = nn.Linear(512, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

    def forward(self, features):
        if self.cfg.MODEL.FC_TO_CONV:
            flattern = self.avg_pool(features)
            nl_ccam_output = self.classifier(flattern)
            nl_ccam_output = nl_ccam_output.view(features.size(0), -1)
        else:
            flattern = self.avg_pool(features).view(features.size(0), -1)
            nl_ccam_output = self.classifier(flattern)
            
        print("nl_ccam_output shape: ",nl_ccam_output.shape)
        # self.nl_ccam_output = nl_ccam_output
        return nl_ccam_output