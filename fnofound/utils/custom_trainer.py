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

def basicLoadModel(model_path: Union[str, Tuple[None, str, Tuple[str]]], _SAVE_LOAD_PARAMS: dict = {}):
    if isinstance(model_path, str):   
        # assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
        model = torch.load(f=model_path, **_SAVE_LOAD_PARAMS)
    else:
        assert isinstance(model_path, tuple) and len(model_path) == 3, \
            'Saving lifting-main part-projection model requires tuple of str arg with len 3.'
        assert isinstance(model_path[1], str), 'Main core path has to be a str.'
        main_fno        = torch.load(f = model_path[1], **_SAVE_LOAD_PARAMS)

        if model_path[0] is None:
            assert (model_path[0] is None), 'Can not load projections without liftings.'
            input_adapters, output_adapters = None, None
        
        elif isinstance(model_path[0], str):
            assert isinstance(model_path[2], str), 'If lifting is passed as a str, proj. has to be a str too.'
            input_adapters  = torch.load(f = model_path[0], **_SAVE_LOAD_PARAMS)
            output_adapters = torch.load(f = model_path[2], **_SAVE_LOAD_PARAMS)
            
        else:
            assert (isinstance(model_path[0], (list, tuple))), \
                'Liftings have to be passed as list or tuple, if multiple adapters are expected.'
            assert (len(model_path[0]) == len(model_path[2])), \
                 f'If liftings are passed as {len(model_path[0])} elems, proj. has to be a {len(model_path[2])} elems.'
            input_adapters, output_adapters = [], []
            for adapter_idx in range(len(model_path[0])):
                input_adapters.append(torch.load(f = model_path[0][adapter_idx], **_SAVE_LOAD_PARAMS))
                output_adapters.append(torch.load(f = model_path[2][adapter_idx], **_SAVE_LOAD_PARAMS))

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

        self.optimizer = set_optimizer(params_opt, self.params_to_optimize)
        self.scheduler = set_scheduler(params_scheduler, self.optimizer)

        if trainer_loss is None:
            self._training_loss = LpLoss(d=n_dim)

    def to(self, device: str = 'cuda'):
        self.device = device

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

    def saveModel(self, model_path: Union[str, Tuple[str, List[str]]]):
        if self._single_model:   
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            torch.save(obj = self.model, f = model_path, **self._SAVE_LOAD_PARAMS)
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            torch.save(obj = self.main_fno,        f = model_path[1], **self._SAVE_LOAD_PARAMS)

            if isinstance(model_path[0], str):
                assert isinstance(model_path[2], str), \
                    'If a string is a path for lifting model, a string has to be a path for proj. too.'
                warnings.warn("Saving a single lifting and projection.")
                torch.save(obj = self.input_adapters[0],  f = model_path[0], **self._SAVE_LOAD_PARAMS)
                torch.save(obj = self.output_adapters[0], f = model_path[2], **self._SAVE_LOAD_PARAMS)

            elif isinstance(model_path[0], (list, tuple)):
                assert (isinstance(model_path[2], (list, tuple)) and len(model_path[0]) == len(model_path[2])), \
                    'If a list/tuple is a path for lifting model, a list/tuple has to be a path for proj. too.'
                assert len(self.input_adapters) == len(model_path[2]), 'Mismatching numbers of filenames and submodels.'
                for idx in range(len(model_path[0])):
                    torch.save(obj = self.input_adapters[idx],  f = model_path[0][idx], **self._SAVE_LOAD_PARAMS)
                    torch.save(obj = self.output_adapters[idx],  f = model_path[2][idx], **self._SAVE_LOAD_PARAMS)
                    

    def loadModel(self, model_path: Union[str, Tuple[None, str, Tuple[str]]]):
        model = basicLoadModel(model_path, self._SAVE_LOAD_PARAMS)
        if isinstance(model, tuple):
            # if isinstance(model, tuple):
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

    # def loadModel(self, model_path):
    #     # Implement loading data from pickle
    #     pass

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
            train_err, val_loss = self.trainSingleEpoch(epoch, train_loader, val_loader, 
                                                          self._training_loss, data_processor)
            print(f'{epoch} - th epoch: train error is {train_err}, val error {val_loss}')

            if train_err < best_err:
                best_err = train_err

            if (epoch == train_epochs-1):
                self.logTraining(train_err, val_loss, 0) # f'Finished model training. train_err: {train_err}, avg_epoch_loss: {avg_epoch_loss}')

        if self._single_model:
            return self.model
        else:
            return self.input_adapters, self.main_fno, self.output_adapters

    def trainSingleEpoch(self, epoch, train_loader: List[DataLoader], val_loader: List[DataLoader], 
                           training_loss, data_processor: List[DataProcessor] = [None,]):
        """trainSingleEpoch trains self.model on train_loader
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
        self.onEpochStart(epoch)

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

                # print('-'*10 + ' For training dataset: ' + '-'*10)
                loss = self.trainOneBatch(epoch, sample, training_loss, data_processor[dataset_idx])
                
                if torch.isnan(loss).item():
                    print('loss is NaN')
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

        # print(f'----------------------------- REACHED VAL -----------------------------')
        with torch.no_grad():
            val_loss = 0.

            n_fine_samples = self.n_samples_val
            # print('-'*10 + ' For validation dataset: ' + '-'*10)

            for dataset_idx, loader in enumerate(val_loader):
                if (isinstance(loader, dict)):
                    for res, resloader in loader.items():
                        for idx, sample in enumerate(resloader):
                            loss = self.trainOneBatch(epoch, sample, training_loss, data_processor[dataset_idx])                    
                            if torch.isnan(loss).item():
                                n_fine_samples -= 1
                                continue
                    
                            val_loss += loss.item()                        
                else:
                    for idx, sample in enumerate(loader):
                        loss = self.trainOneBatch(epoch, sample, training_loss, data_processor[dataset_idx])                    
                        if torch.isnan(loss).item():
                            n_fine_samples -= 1
                            continue
                
                        val_loss += loss.item()

            val_loss /=  n_fine_samples


        self.logTraining(val_loss=val_loss, train_err=train_err, lr=lr)

        return train_err, val_loss # , avg_lasso_loss

    def logTraining(self, val_loss, train_err, lr): # epoch,
        self._logger.write({'val_loss': val_loss, 'train_err': train_err, 'lr': lr}) # 'Epoch': epoch, 
        # self._logger.write('Epoch: {} | avg_loss: {} | train_err: {} | lr: {}'.format(epoch, avg_loss, train_err, lr))


    def onEpochStart(self, *args, **kwargs):
        """
        Stub for implementing additional logick!
        """
        pass

    def trainOneBatch(self, idx, sample, training_loss, data_processor = None): # , train_mode = True
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
        HEATMAPS = False; HMP_idx = 5

        sample["x"] = sample["x"].to(self.device)
        sample["y"] = sample["y"].to(self.device)
        if "mask" in sample.keys():
            sample["mask"] = sample["mask"].to(self.device)
        
        if idx == HMP_idx and HEATMAPS:    
            print(f'sample.shape is {sample["x"].shape} - {sample["y"].shape}')
            for channel in range(sample['x'].shape[1]):
                Heatmap(sample['x'][0, channel, -5, ...].cpu().detach().numpy(), title=f'Input: channel {channel} before preprocess')

            for channel in range(sample['y'].shape[1]):
                Heatmap(sample['y'][0, channel, -5, ...].cpu().detach().numpy(), title=f'Reference: channel {channel} before preprocess')

        if data_processor is not None:
            if isinstance(sample, dict):
                sample = data_processor.preprocess(sample) # X: 
            else:
                warnings.warn('Possibly, incorrect type of model input')

        if idx == HMP_idx and HEATMAPS:    
            print(f'sample.shape is {sample["x"].shape} - {sample["y"].shape}')
            for channel in range(sample['x'].shape[1]):
                Heatmap(sample['x'][0, channel, -5, ...].cpu().detach().numpy(), title=f'Input: channel {channel}')

            for channel in range(sample['y'].shape[1]):
                Heatmap(sample['y'][0, channel, -5, ...].cpu().detach().numpy(), title=f'Reference: channel {channel}')

        # X, Y, eq_idx = sample  # X: [B, CX, T, X ...], Y: [B, CY, T, X ...]

        # print('X', X, 'Y', Y, 'eq_idx', eq_idx)
        # if self.regularizer:
        #     self.regularizer.reset()
        
        # if data_processor is not None:
            # X = sample["x"] # data_processor.preprocess(X) # X: 
            
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
                    out = self.model(sample["x"])
                else:
                    out = self.input_adapters[sample["eq_idx"][0].item()](sample["x"])
                    out = self.main_fno(out)
                    out = self.output_adapters[sample["eq_idx"][0].item()](out)

        else:
            if self._single_model:
                out = self.model(sample["x"])
            else:
                out = self.input_adapters[sample["eq_idx"][0].item()](sample["x"])
                out = self.main_fno(out)
                out = self.output_adapters[sample["eq_idx"][0].item()](out)
        
        # if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
        #     print(f"Raw outputs of shape {out.shape}")

        # print(f'In trainOneBatch: before preprocessor {torch.mean(out), torch.std(out)}')
        
        if idx == HMP_idx and HEATMAPS:    
            for channel in range(out.shape[1]):
                Heatmap(out[0, channel, -5, ...].cpu().detach().numpy(), title=f'Model output: channel {channel} before norm')

        if data_processor is not None:
            out, sample = data_processor.postprocess(out, sample)

        # loss = 0.0

        # print(f'Obtained out mean - {torch.mean(out)}, ref mean - {torch.mean(X)}')

        if idx == HMP_idx and HEATMAPS:    
            for channel in range(out.shape[1]):
                Heatmap(out[0, channel, -5, ...].cpu().detach().numpy(), title=f'Model output: channel {channel}')
                Heatmap(torch.abs(out[0, channel, -5, ...] - sample["y"][0, channel, -5, ...]).cpu().detach().numpy(),
                        title = f'Diff.: channel {channel}') 
                # torch.nn.functional.mse_loss(out, X) #

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss = training_loss(out, sample["y"]) # torch.nn.functional.mse_loss(out, X) #
        else:
            # loss += torch.nn.functional.mse_loss(out, Y) # training_loss(out, X)
            # print(f'shapes for loss {sample["x"].shape} -> {out.shape} & {sample["y"].shape}')
            loss = training_loss(out, sample["y"])
        
        return loss


    def finetune():
        pass
