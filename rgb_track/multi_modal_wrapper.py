import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import combinations
import numpy as np
from simplenet import SimpleNet112
from transformer import TransformerEncoder

def get_backbone(architecture_name, 
                 pretrained=None, 
                 get_feature_size=False,
                 in_channels=3):
    feature_size = None 
    model = SimpleNet112(pretrained=pretrained, in_channels=in_channels)
    feature_size = 256
   
    if get_feature_size:
        return model, feature_size
    else:
        return model
        
class MultiModalWrapper(nn.Module):
    def __init__(self, wrapper_config):
        print('multi __init__')
        super().__init__()
        self.backbone = None
        self.classifier = None
        self._init_modules(wrapper_config)
        self._init_loss(wrapper_config)
        self.batch_info = {}

    def _init_modules(self, wrapper_config):
        self.input_modalities = wrapper_config.input_modalities
        for modal_key in self.input_modalities:
            if (modal_key == 'optical_flow') or (modal_key == 'optical_flow_start'):
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=wrapper_config.pretrained,
                                                      get_feature_size=True,
                                                      in_channels=2)
            else:
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=wrapper_config.pretrained,
                                                      get_feature_size=True)

            setattr(self, 'backbone_' + modal_key, backbone)

        self.backbone_feature_size = feature_size
        self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        self.pooling2 = nn.AdaptiveMaxPool2d((1, feature_size))
        self.pooling3 = nn.AdaptiveMaxPool2d((1, feature_size))

        self.classifier = nn.Sequential(
            nn.Linear(3 * feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, wrapper_config.nclasses)
        )
        self.classifier = nn.Linear(3 * feature_size, wrapper_config.nclasses)

    def _init_loss(self, wrapper_config):
        loss_config = None
        if hasattr(wrapper_config, 'loss_config'):
            loss_config = wrapper_config.loss_config
        self.loss = FocalLoss.get_loss(wrapper_config.loss, loss_config)


    def forward(self, x):
        print('multi forward')
        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        M = len(self.input_modalities)
        features = torch.empty((B, M, self.backbone_feature_size)).to(device)

        for idx, key in enumerate(self.input_modalities):
            features[:, idx, :] = getattr(self, 'backbone_' + key)(x[key])
        features = features.view((B, M, -1))

        features1 = self.pooling(features)
        features2 = self.pooling2(features)
        features3 = self.pooling3(-features)
        features = torch.cat([features1, features2, features3], axis=2)
        features = features.squeeze()
        output = self.classifier(features)
        sigmoid_output = torch.sigmoid(output)
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        targets = targets.type(outputs.type())

        logpt = -F.binary_cross_entropy_with_logits(
            outputs, targets, reduction="none"
        )
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()

        return loss

    @staticmethod
    def get_loss(loss_name, loss_config=None):
        if loss_name == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        elif loss_name == 'CCE':
            loss = nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            gamma = getattr(loss_config, 'gamma', 2.0)
            alpha = getattr(loss_config, 'alpha', 0.25)
            loss = FocalLoss(gamma=gamma,
                            alpha=alpha)
        else:
            raise Exception('Unknown loss type')
        return loss
