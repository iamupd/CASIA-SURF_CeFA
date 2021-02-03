import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import combinations
import numpy as np
from transformer import TransformerEncoder
import ipdb

class MultiModalWrapper(nn.Module):
    def __init__(self, wrapper_config):
        print('multi __init__')
        super().__init__()
        self.backbone = None
        self.classifier = None
        self.loss = nn.BCEWithLogitsLoss()
        self.batch_info = {}
        self.backbone_feature_size = 256
        self.input_modalities = wrapper_config.input_modalities

        self.backbone_optical_flow = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False, stride=1), # 16x110x110
            nn.BatchNorm2d(16), 
            nn.ReLU(),  
            nn.MaxPool2d(2),                                        # 16x55x55

            nn.Conv2d(16, 32, kernel_size=3, bias=False, stride=1), # 32x53x53
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32x26x26
            
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1), # 64x24x24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 64x12x12

            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=1),# 128x10x10
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 128x5x5
            nn.Conv2d(128, 256, kernel_size=5))                     # 256x1

        
        self.backbone_optical_flow_start = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5))

        self.backbone_stat_r1000 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5))

        self.backbone_stat_r1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5)) 

        self.pooling = nn.AdaptiveAvgPool2d((1, self.backbone_feature_size))
        self.pooling2 = nn.AdaptiveMaxPool2d((1, self.backbone_feature_size))
        self.pooling3 = nn.AdaptiveMaxPool2d((1, self.backbone_feature_size))

        self.classifier = nn.Sequential(
            nn.Linear(3 * self.backbone_feature_size, self.backbone_feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_feature_size, wrapper_config.nclasses)
        )
        self.classifier = nn.Linear(3 * self.backbone_feature_size, wrapper_config.nclasses)

    def forward(self, x):
        print('multi forward')
        if len(x[self.input_modalities[0]].size()) == 3:
            for key in self.input_modalities:
                x[key] = torch.unsqueeze(x[key], 0)

        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        M = len(self.input_modalities)
        features = torch.empty((B, M, self.backbone_feature_size)).to(device)

    #   out = out.view(out.size(0), -1))
 #       for idx, key in enumerate(self.input_modalities): 
 #           features[:, idx, :] = getattr(self, 'backbone_' + key)(x[key])
        features[:, 0, :] = self.backbone_optical_flow(x['optical_flow']).view(256, -1)
        features[:, 1, :] = self.backbone_optical_flow_start(x['optical_flow_start']).view(256, -1)
        features[:, 2, :] = self.backbone_stat_r1000(x['stat_r1000']).view(256, -1)
        features[:, 3, :] = self.backbone_stat_r1(x['stat_r1']).view(256, -1)

        features = features.view((B, M, -1))

        features1 = self.pooling(features)
        features2 = self.pooling2(features)
        features3 = self.pooling3(-features)
        features = torch.cat([features1, features2, features3], axis=2)
        features = features.squeeze()
        output = self.classifier(features)
        sigmoid_output = torch.sigmoid(output)
    #    if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
    #        x['target'] = x['target'].squeeze()

        output_dict = {'output': sigmoid_output.detach().cpu().numpy(),
                       'target': x['target'].detach().cpu().numpy()}
        for k, v in x.items():
            if k not in ['data', 'target'] + self.input_modalities:
                output_dict[k] = v

        loss = self.loss(output, x['target'])
        return output_dict, loss

    def predict(self, x):
        features = self.backbone(x['data'])
        output = self.classifier(features)
        output_dict = {'output': output.detach().cpu()}
        return output_dict

    def to_parallel(self, parallel_class):
        self.backbone = parallel_class(self.backbone)
        self.classifier = parallel_class(self.classifier)
        return self
