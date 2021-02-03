import torch
import torch.utils.data
from casia_video_dataset import VideoDataset

class DatasetManager(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_sampler(sampler_config, dataset):
        if type(sampler_config.num_elem_per_epoch) == int:
            num_elements = sampler_config.num_elem_per_epoch
        elif type(sampler_config.num_elem_per_epoch) == float:
            num_elements = int(sampler_config.num_elem_per_epoch * len(dataset))

        if num_elements > len(dataset):
            replacement = True
        else:
            replacement = False
            num_elements = None
        sampler = torch.utils.data.sampler.RandomSampler(dataset,
                                                            replacement,
                                                            num_elements)
        return sampler

    @staticmethod
    def get_dataloader(dataset_config, train_process_config, shuffle=True):
        dataset = VideoDataset(dataset_config, dataset_config.transforms)
        if hasattr(dataset_config, 'sampler_config'):
            sampler = DatasetManager._get_sampler(dataset_config.sampler_config, dataset)
            shuffle = False
        else:
            sampler = None
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_process_config.batchsize,
                                                  shuffle=shuffle,
                                                  num_workers=train_process_config.nthreads,
                                                  sampler=sampler)
        return data_loader

    @staticmethod
    def get_dataloader_by_args(dataset, batch_size, num_workers=8, shuffle=False, sampler=None):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  sampler=sampler)
        return data_loader
