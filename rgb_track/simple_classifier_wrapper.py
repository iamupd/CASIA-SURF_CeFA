import torch.nn as nn
from simplenet import SimpleNet112

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

class SimpleClassifierWrapper(nn.Module):
    def __init__(self, wrapper_config):
        print('simple __init__')
        super().__init__()
        self.backbone = None
        self.classifier = None
        self._init_modules(wrapper_config)
        self._init_loss(wrapper_config)
        self.batch_info = {}

    def _init_modules(self, wrapper_config):
        self.backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                   pretrained=wrapper_config.pretrained,
                                                   get_feature_size=True)
        self.classifier = nn.Linear(feature_size, wrapper_config.nclasses)

    def _init_loss(self, wrapper_config):
        loss_config = None
        if hasattr(wrapper_config, 'loss_config'):
            loss_config = wrapper_config.loss_config
        self.loss = FocalLoss.get_loss(wrapper_config.loss, loss_config)

    def forward(self, x):
        print('simple forward')
        features = self.backbone(x['data'])
        output = self.classifier(features)
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()
        output_dict = {'output': output.detach().cpu().numpy(),
                       'target': x['target'].detach().cpu().numpy()}
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

    def to(self, device):
        for attribute, attr_value in self.__dict__['_modules'].items():
            print('wrapper ', attribute)
            if isinstance(attr_value, nn.Module):
                setattr(self.__dict__['_modules'], attribute, attr_value.to(device))
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
