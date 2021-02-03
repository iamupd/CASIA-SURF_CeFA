#from at_learner_core.trainer import Runner
import time
import os
import torch
from collections import OrderedDict
from dataset_manager import DatasetManager
from utils.optimizer import get_lr_scheduler
from utils.optimizer import get_optimizer
import loggers
import utils
from multi_modal_wrapper import MultiModalWrapper
import torch.optim as optim


class State(object):
    """
    A class used to represent a state of model

    ...

    Attributes
    ----------
    root : Model
        Training model object.
    config : argparse.Namespace
        Training process config
    state : dict
        State dictionary to load and continue the training process.
        State includes epoch number, wrapper state dict, lr_scheduler, optimizer params etc.

    Methods
    -------
    create()
        Create self.state dictionary
    save()
        Save state
    save_checkpoint(filename)
        Save checkpoint to *filename*
    load_checkpoint()
        load checkpoint from .pth and set self.root attributes
    """
    def __init__(self, root):
        self.root = root
        self.config = self.root.config
        self.root_dir = self.config.checkpoint_config.out_path
        self.state = None

    def create(self):
        # Params to be saved in checkpoint
        self.state = {
            'epoch': self.root.epoch,
            'state_dict': self.root.wrapper.state_dict(),
            'lr_scheduler': self.root.lr_scheduler.state_dict(),
            'optimizer': self.root.optimizer.state_dict(),
        }

    def save(self):
        if self.config.checkpoint_config.save_frequency == 0:
            self.save_checkpoint('checkpoint.pth')
        else:
            if self.root.epoch % self.config.checkpoint_config.save_frequency == 0:
                self.save_checkpoint('model_{}.pth'.format(self.root.epoch))

    def save_checkpoint(self, filename):  # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.root_dir, 'checkpoints', filename)
        torch.save(self.state, fin_path)

    def load_checkpoint(self):  # Load current checkpoint if exists
        fin_path = os.path.join(self.root_dir, 'checkpoints', self.root.config.resume)
        if os.path.isfile(fin_path):
            print(">>>> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location='cpu')
            self.root.epoch = checkpoint['epoch'] + 1
            self.root.model.load_state_dict(checkpoint['state_dict'])
            self.root.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.root.optimizer.load_state_dict(checkpoint['optimizer'])

            print(">>>> loaded checkpoint '{}' (epoch {})".format(self.root.config.resume, checkpoint['epoch']))
        else:
            print(">>>> no checkpoint found at '{}'".format(self.root.config.resume))


class RGBRunner(object):
    def __init__(self, config, train=True):
        self.config = config
        self.device = torch.device("cuda" if config.train_process_config.ngpu else "cpu")

        self._init_wrapper()
        if train:
            """
            TODO:
            * add parameters, optimizer and scheduler to session iterations
            * add get_optimizer and get_lr_scheduler to 'models' directory. I need discuss.
            * add utils.Metrics with self.config 
            """
            self._init_optimizer()
            self._init_logger()
            self.state = State(self)
            self._init_metrics()
            self._init_loaders()
            self.epoch = 0
            self.best_epoch = False
            self.training = False

    def _init_wrapper(self):
        self.wrapper = MultiModalWrapper(self.config.wrapper_config)
        self.wrapper = self.wrapper.to(self.device)

    def _init_optimizer(self):
        parameters = [{'params': self.wrapper.parameters()}]
        self.optimizer = optim.Adam(parameters,
                               lr=0.0001,
                               betas=(0.9, 0.999),
                               weight_decay=1e-05)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                 step_size=5,
                                                 gamma=0.5)

    def _init_metrics(self):
        self.train_info = utils.LossMetricsMeter(self.config.test_process_config)
        self.test_info = utils.LossMetricsMeter(self.config.test_process_config)
        self.best_test_info = utils.meters.AverageMeter()
        self.batch_time = utils.meters.AverageMeter()
        return

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False)
    def _process_on_epoch_start(self):
        """
        This method was created for preprocessing before the beginning of epoch.
        For example, change dataset indices.
        :return:
        """
        pass

    def _init_logger(self):
        self.logger = loggers.get_logger(self)
  
    def _check_best_epoch(self):
        self.best_epoch = False


    def train(self):
        if self.config.resume:
            self.state.load_checkpoint()

        if self.config.train_process_config.ngpu > 1:
            self.wrapper.to_parallel(torch.nn.DataParallel)
        """
        TODO:
        * Add iteration by sessions
        * Add freeze parameters
        """
        for epoch in range(self.epoch, self.config.train_process_config.nepochs):
            self.epoch = epoch
            self._process_on_epoch_start()
            self._train_epoch()
            self.lr_scheduler.step()

            if hasattr(self.config.test_process_config, 'run_frequency'):
                if self.config.test_process_config.run_frequency == -1:
                    pass
                elif epoch % self.config.test_process_config.run_frequency == 0:
                    self._test_epoch()
            else:
                self._test_epoch()

            self._check_best_epoch()
            self.logger.log_epoch()
            self.state.create()
            if self.best_epoch:
                self.state.save_checkpoint('best_model_{epoch}.pth'.format(epoch=str(self.epoch).zfill(4)))
            else:
                self.state.save()

    def _train_epoch(self):
        self.wrapper.train()
        self.training = True
        self.train_info.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            if isinstance(data, dict) or isinstance(data, OrderedDict):
                for k, v in data.items():
                    if isinstance(v, list):
                        data[k] = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        if isinstance(v, torch.Tensor):
                            data[k] = v.to(self.device)
            else:
                data = data.to(self.device)

            self.optimizer.zero_grad()
            output_dict, batch_loss = self.wrapper(data)  # TODO
            batch_loss.backward()
            self.optimizer.step()
            
            self.train_info.update(batch_loss, output_dict)
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.logger.log_batch(batch_idx)

    def _test_epoch(self):
        self.wrapper.eval()
        self.training = False
        self.batch_time.reset()
        self.test_info.reset()
        time_stamp = time.time()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                if isinstance(data, dict) or isinstance(data, OrderedDict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            data[k] = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in v]
                        else:
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(self.device)
                else:
                    data = data.to(self.device)

                output_dict, batch_loss = self.wrapper(data)
                                
                self.test_info.update(batch_loss, output_dict)
                self.batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()
                self.logger.log_batch(batch_idx)
