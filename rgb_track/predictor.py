from collections import OrderedDict
from tqdm import tqdm
import os
import torch
import utils
import loggers
from dataset_manager import DatasetManager
from casia_video_dataset import VideoDataset
import torch.nn as nn
from simplenet import SimpleNet112
from multi_modal_wrapper import MultiModalWrapper
import ipdb

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
        self.wrapper = MultiModalWrapper(self.model_config.wrapper_config)
        self.wrapper.load_state_dict(checkpoint['state_dict'])
        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        dataset_config = self.test_config.dataset_configs
        transforms = self.model_config.datalist_config.testlist_configs.transforms
        setattr(dataset_config, 'transforms', transforms)

        seq_transforms = self.model_config.datalist_config.testlist_configs.sequence_transforms
        setattr(dataset_config, 'sequence_transforms', seq_transforms)
    #    dataset = VideoDataset(dataset_config, dataset_config.transforms)
    #    self.test_loader = torch.utils.data.DataLoader(dataset,
    #                                              batch_size=dataset_config.batch_size,
    #                                              num_workers=dataset_config.nthreads)
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
                
                print(data.keys())
                output_dict, batch_loss = self.wrapper(data)

                self.test_info.update(batch_loss, output_dict)
                self.logger.log_batch(batch_idx)

        self.logger.log_epoch()
        self.logger.close()


    def run_predict_one(self, img):
        self.wrapper.eval()
        item_dict = OrderedDict()
        item_dict['data'] = img
        item_dict['target'] = 0
        with torch.no_grad():
            item_dict['target'] = torch.Tensor([item_dict['target']])
            item_dict = self.test_config.dataset_configs.transforms(item_dict)

            if isinstance(item_dict, dict) or isinstance(item_dict, OrderedDict):
                for k, v in item_dict.items():
                    if isinstance(v, torch.Tensor):
                        item_dict[k] = v.to(self.device)

            else:
                item_dict = item_dict.to(self.device)

            item_dict['optical_flow'] = torch.unsqueeze(item_dict['optical_flow'], 0)
            item_dict['optical_flow_start'] = torch.unsqueeze(item_dict['optical_flow_start'], 0)
            item_dict['stat_r1000'] = torch.unsqueeze(item_dict['stat_r1000'], 0)
            item_dict['stat_r1'] = torch.unsqueeze(item_dict['stat_r1'], 0)

            output_dict, batch_loss = self.wrapper(item_dict)
        return output_dict['output'].item()