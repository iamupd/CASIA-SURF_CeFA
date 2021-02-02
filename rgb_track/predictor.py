from collections import OrderedDict
from tqdm import tqdm
import os
import torch
#from . import models
import utils
import loggers
from dataset_manager import DatasetManager
from simple_classifier_wrapper import SimpleClassifierWrapper
from casia_video_dataset import VideoDataset
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


class Predictor(object):
    def __init__(self, test_config, model_config, checkpoint_path):
        self.test_config = test_config
        self.model_config = model_config
        self.device = torch.device("cuda" if self.test_config.ngpu else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.epoch = '.'.join(os.path.basename(checkpoint_path).split('_')[-1].split('.')[:-1])

        self._init_wrapper(checkpoint)
        self._init_loaders()
        self._init_metrics()
        self._init_logger()

    def _init_logger(self):
        logger_config = self.test_config.logger_config
        self.logger = loggers.get_logger(self, logger_config)

    def _init_wrapper(self, checkpoint):
        self.wrapper = models.get_wrapper(self.model_config)
        self.wrapper.load_state_dict(checkpoint['state_dict'])
        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        dataset_config = self.test_config.dataset_configs
        if dataset_config.transform_source == 'model_config':
            transforms = self.model_config.datalist_config.testlist_configs.transforms
            setattr(dataset_config, 'transforms', transforms)

        if dataset_config.seq_transform_source == 'model_config':
            seq_transforms = self.model_config.datalist_config.testlist_configs.sequence_transforms
            setattr(dataset_config, 'sequence_transforms', seq_transforms)
        dataset = VideoDataset(dataset_config, dataset_config.transforms)
        self.test_loader = DatasetManager.get_dataloader_by_args(dataset=dataset,
                                                                 batch_size=dataset_config.batch_size,
                                                                 num_workers=dataset_config.nthreads)

    def _init_metrics(self):
        self.test_info = utils.LossMetricsMeter(self.test_config.dataset_configs.test_process_config)

    def run_predict(self):
        self.wrapper.eval()
        self.test_info.reset()
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.test_loader),
                                        total=len(self.test_loader)):
                if isinstance(data, dict) or isinstance(data, OrderedDict):
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            data[k] = v.to(self.device)
                else:
                    data = data.to(self.device)

                output_dict, batch_loss = self.wrapper(data)

                self.test_info.update(batch_loss, output_dict)
                self.logger.log_batch(batch_idx)

        self.logger.log_epoch()
        self.logger.close()
