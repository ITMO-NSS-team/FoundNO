import os
import warnings
import logging
import json

import math

from functools import singledispatchmethod
from typing import Tuple, List, Union

import numpy as np
import torch
import torch.distributed as dist

from neuralop.data.transforms.data_processors import DataProcessor

from .data_utils import Dataset, Heatmap
from torch.utils.data import DataLoader

from .logger import Logger
from .optimizer_utils import set_optimizer, set_scheduler
from .training_utils import LpLoss

def load_model(model_path: Union[str, Tuple[str]], _SAVE_LOAD_PARAMS: dict = {}):
    if isinstance(model_path, str):   
        # assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
        model = torch.load(f=model_path, **_SAVE_LOAD_PARAMS)
    else:
        assert isinstance(model_path, tuple) and len(model_path) == 3, \
            'Saving lifting-main part-projection model requires tuple of str arg with len 3'
        input_adapters  = torch.load(f = model_path[0], **_SAVE_LOAD_PARAMS)
        main_fno        = torch.load(f = model_path[1], **_SAVE_LOAD_PARAMS)
        output_adapters = torch.load(f = model_path[2], **_SAVE_LOAD_PARAMS)
        model = (input_adapters, main_fno, output_adapters)

    return model


class Trainer(object):
    mixed_precision = False # Load them from param json
    verbose = False
    eval_interval = 1000

    _SAVE_LOAD_PARAMS = {}

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError('Due to high expected load, only cuda must be supported.')
        self.device = 'cuda' # Using NCCL backend

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
        else:
            self.world_size = 1

        if self.world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self.local_rank = 1
            self.world_rank = 1

        self.input_experts = None
        self.main_fno = None

    @singledispatchmethod
    def buildModel(self, model):
        raise NotImplementedError("Cannot declare model with anything, but nn Module or tuple of nn Modules")
    
    @buildModel.register
    def _(self, model: torch.nn.Module):
        self._single_model = True
        self.model = model
        self.params_to_optimize = [{'params': self.model.parameters()},]

    @buildModel.register
    def _(self, model: tuple): #expect Tuple[List[torch.nn.Module], torch.nn.Module, List[torch.nn.Module]]
        assert len(model) == 3, \
            'Multiple adapter architecture requires sequence of input adapters -> single model -> output adapters'

        assert isinstance(model[0], list) and isinstance(model[0][0], torch.nn.Module), \
            'Liftings have to be set as a list of torch nn Modules'

        assert isinstance(model[1], torch.nn.Module), \
            'Main neural operator model has to be set as a single torch nn Module'
        
        assert isinstance(model[2], list) and isinstance(model[2][0], torch.nn.Module), \
            'Projections have to be set as a list of torch nn Modules'

        assert len(model[0]) == len(model[2]), 'Numbers of liftings and projections have to match.'

        self._single_model = False
        self.input_adapters = model[0]
        self.main_fno = model[1]
        self.output_adapters = model[2]

        self.params_to_optimize = []
        for idx_expert_nn, _ in enumerate(self.input_adapters):
            self.params_to_optimize.append({'params': self.input_adapters[idx_expert_nn].parameters()})
            self.params_to_optimize.append({'params': self.output_adapters[idx_expert_nn].parameters()})

    def buildOptimizer(self, 
                        n_dim: int,
                        params_scheduler: dict,
                        params_opt: dict,
                        trainer_loss = None):
        assert self.params_to_optimize is not None, 'Optimizer has to be constructed only after model declaration.'
        # self.data_processor = data_processor

        self.optimizer = set_optimizer(params_opt, self.params_to_optimize)
        self.scheduler = set_scheduler(params_scheduler, self.optimizer)

        if trainer_loss is None:
            self._training_loss = LpLoss(d=n_dim)

    def to(self, device: str = 'cuda'):
        if self._single_model:
            self.model.to(device)
        else:
            if self.main_fno is None or self.input_adapters is None or self.output_adapters is None:
                raise AttributeError('Hidden Fourier NO layers and projection or liftings are not yet declared.')
            
            self.main_fno.to(device)
            for idx, _ in enumerate(self.input_adapters):
                self.input_adapters[idx].to(device)
                self.output_adapters[idx].to(device)

    def setLogger(self, filename, logger: Logger = None, log_level = logging.INFO, logger_name: str = 'FoundationalFNO'):
        if logger is None:
           self._logger = Logger(filename = filename, log_level = log_level, logger_name = logger_name, 
                                 write_every = 1e1, epochs_aggreg = 5, 
                                 info_entries = ['val_loss', 'train_err', 'lr']) # 'Epoch', 
        else:
            self._logger = logger

    def saveModel(self, model_path: Union[str, Tuple[str]]):
        if self._single_model:   
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            torch.save(obj = self.model, f = model_path, **self._SAVE_LOAD_PARAMS)
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            torch.save(obj = self.input_adapters,  f = model_path[0], **self._SAVE_LOAD_PARAMS)
            torch.save(obj = self.main_fno,        f = model_path[1], **self._SAVE_LOAD_PARAMS)
            torch.save(obj = self.output_adapters, f = model_path[2], **self._SAVE_LOAD_PARAMS)

    def loadModel(self, model_path: Union[str, Tuple[str]]):
        model = load_model(model_path, self._SAVE_LOAD_PARAMS)
        if isinstance(model, tuple):
            self.input_adapters = model[0]
            self.main_fno = model[1]
            self.output_adapters = model[2]
        else:
            self.model = model

        # if self._single_model:   
        #     assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
        #     self.model = torch.load(f=model_path, **self._SAVE_LOAD_PARAMS)
        # else:
        #     assert isinstance(model_path, tuple) and len(model_path) == 3, \
        #         'Saving lifting-main part-projection model requires tuple of str arg with len 3'
        #     self.input_adapters     = torch.load(f = model_path[0], **self._SAVE_LOAD_PARAMS)
        #     self.fno_and_proj_model = torch.load(f = model_path[1], **self._SAVE_LOAD_PARAMS)
        #     self.output_adapters    = torch.load(f = model_path[2], **self._SAVE_LOAD_PARAMS)

    def loadModel(self, model_path):
        # Implement loading data from pickle
        pass

    def loadData(self, file):
        pass

    def train(self, train_loader: Union[DataLoader, list], val_loader: Union[DataLoader, list], 
              train_epochs: int, data_processor: Union[list, DataProcessor] = None):
        if isinstance(train_loader, DataLoader):
            train_loader = [train_loader,]
        if isinstance(val_loader, DataLoader):
            val_loader = [val_loader,]
        
        # track number of training examples in batch
        self.n_samples = sum([len(loader) for loader in train_loader]) # .size
        self.n_samples_val = sum([len(loader) for loader in val_loader])

        if isinstance(data_processor, DataProcessor):
            data_processor = [data_processor,]
        elif data_processor is None:
            data_processor = [None,]

        best_err = np.inf

        if self._single_model:
            n_params = sum(p.numel() for p in self.model.parameters())
            init_log = 'Initializing training of model of type' + \
                       ' {} | epochs: {} | n params: {}'.format(type(self.model),
                                                                train_epochs,
                                                                n_params)
        else:
            n_params = sum(p.numel() for p in self.input_adapters[0].parameters()) + \
                       sum(p.numel() for p in self.main_fno.parameters()) + \
                       sum(p.numel() for p in self.output_adapters[0].parameters())

            init_log = 'Initializing training of model of type' + \
                       ' {}, {}, {} | epochs: {} | n params: {}'.format(type(self.input_adapters[0]),
                                                                        type(self.main_fno),
                                                                        type(self.output_adapters[0]),
                                                                        train_epochs,
                                                                        n_params)
        self._logger.write(init_log)

        for epoch in range(train_epochs):
            train_err, val_loss = self.train_single_epoch(epoch, train_loader, val_loader, 
                                                          self._training_loss, data_processor)
            print(f'{epoch} - th epoch: train error is {train_err}, val error {val_loss}')

            if train_err < best_err:
                best_err = train_err

            if (epoch == train_epochs-1):
                self.log_training(train_err, val_loss, 0) # f'Finished model training. train_err: {train_err}, avg_epoch_loss: {avg_epoch_loss}')

        if self._single_model:
            return self.model
        else:
            return self.input_adapters, self.main_fno, self.output_adapters

    def train_single_epoch(self, epoch, train_loader: List[DataLoader], val_loader: List[DataLoader], 
                           training_loss, data_processor: List[DataProcessor] = [None,]):
        """train_single_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : subclass of torch.utils.data.DataLoader
            data loader of train examples

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)

        if self._single_model:
            self.model.train()
        else:
            self.main_fno.train()

            for idx, _ in enumerate(self.input_adapters):
                self.input_adapters[idx].train()
                self.output_adapters[idx].train()


        if data_processor[0] is not None:
            for idx in range(len(data_processor)):
                data_processor[idx].train()

        if self._single_model:
            self.model.train()
        else:
            for idx_adapter, _ in enumerate(self.input_adapters):
                self.input_adapters[idx_adapter].train()
                self.output_adapters[idx_adapter].train()
            self.main_fno.train()

        train_err = 0.0
        n_fine_samples = self.n_samples
        for dataset_idx, loader in enumerate(train_loader):
            for idx, sample in enumerate(loader):
                
                self.optimizer.zero_grad()

                loss = self.train_one_batch(idx, sample, training_loss, data_processor[dataset_idx])
                
                if torch.isnan(loss).item():
                    n_fine_samples -= 1
                    continue
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    train_err += loss.item()

        # print('n_fine_samples for training', n_fine_samples)
        train_err /= n_fine_samples

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]

        if self._single_model:
            self.model.eval()
        else:
            for idx_adapter, _ in enumerate(self.input_adapters):
                self.input_adapters[idx_adapter].eval()
                self.output_adapters[idx_adapter].eval()
            self.main_fno.eval()

        with torch.no_grad():
            val_loss = 0.

            n_fine_samples = self.n_samples_val

            for dataset_idx, loader in enumerate(val_loader):
                for idx, sample in enumerate(loader):
                    loss = self.train_one_batch(idx, sample, training_loss, data_processor[dataset_idx])                    
                    if torch.isnan(loss).item():
                        n_fine_samples -= 1
                        continue
            
                    val_loss += loss.item()

            val_loss /=  n_fine_samples


        self.log_training(val_loss=val_loss, train_err=train_err, lr=lr)

        return train_err, val_loss # , avg_lasso_loss

    def log_training(self, val_loss, train_err, lr): # epoch,
        self._logger.write({'val_loss': val_loss, 'train_err': train_err, 'lr': lr}) # 'Epoch': epoch, 
        # self._logger.write('Epoch: {} | avg_loss: {} | train_err: {} | lr: {}'.format(epoch, avg_loss, train_err, lr))


    def on_epoch_start(self, *args, **kwargs):
        """
        Stub for implementing additional logick!
        """
        pass

    def train_one_batch(self, idx, sample, training_loss, data_processor = None): # , train_mode = True
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : tuple(torch.Tensor, torch.Tensor, int)
            data tuple holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        X, Y, eq_idx = sample 

        # if self.regularizer:
        #     self.regularizer.reset()
        
        if data_processor is not None:
            X = data_processor.preprocess(X)
            
        # else:
        #     # load data to device if no preprocessor exists
            
        #     sample = {
        #         k: v.to(self.device)
        #         for k, v in sample.items()
        #         if torch.is_tensor(v)
        #     }

        # if isinstance(Y, torch.Tensor):
        #     self.n_samples += Y.shape[0]
        # else:
        #     self.n_samples += 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                if self._single_model:
                    out = self.model(X)
                else:
                    out = self.input_adapters[eq_idx[0].item()](X)
                    out = self.main_fno(out)
                    out = self.output_adapters[eq_idx[0].item()](out)

        else:
            if self._single_model:
                out = self.model(X)
            else:
                out = self.input_adapters[eq_idx[0].item()](X)
                out = self.main_fno(out)
                out = self.output_adapters[eq_idx[0].item()](out)
        
        # if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
        #     print(f"Raw outputs of shape {out.shape}")

        if data_processor is not None:
            out, Y = data_processor.postprocess(out, Y)

        # loss = 0.0

        # print(f'Obtained out mean - {torch.mean(out)}, ref mean - {torch.mean(X)}')

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss = training_loss(out, Y) # torch.nn.functional.mse_loss(out, X) #
        else:
            # loss += torch.nn.functional.mse_loss(out, Y) # training_loss(out, X)

            loss = training_loss(out, Y)

        # Heatmap(out[0, 0, 10, ...].cpu().detach().numpy())
        # del X, Y, out
        # torch.cuda.empty_cache()           
        # Heatmap(out[0, 0, ...].cpu().detach().numpy())
        
        return loss


    def finetune():
        pass