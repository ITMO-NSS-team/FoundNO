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

from .data_utils import Dataset

from .logger import Logger
from .optimizer_utils import set_optimizer, set_scheduler
from .training_utils import LpLoss

class FineTuner(object):
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
        self.fno_and_proj_model = None

    @singledispatchmethod
    def buildModel(self, model):
        raise NotImplementedError("Cannot declare model with anything, but nn Module or tuple of nn Modules")
    
    @buildModel.register
    def _(self, model: torch.nn.Module):
        warnings.warn('Building fine-tuning trainer with a single model is not advised.')
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
        
        assert isinstance(model[2], list) and isinstance(model[2][0], list), \
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
                        data_processor = None,
                        trainer_loss = None):
        assert self.params_to_optimize is not None, 'Optimizer has to be constructed only after model declaration.'
        self.data_processor = data_processor

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
           self._logger = Logger(filename = filename, log_level = log_level, logger_name = logger_name)
        else:
            self._logger = logger

    def saveModel(self, model_path: Union[str, Tuple[str]]):
        if self._single_model:   
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            torch.save(obj = self.model, f = model_path, **self._SAVE_LOAD_PARAMS)
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            torch.save(obj = self.input_adapters,     f = model_path[0], **self._SAVE_LOAD_PARAMS)
            torch.save(obj = self.fno_and_proj_model, f = model_path[1], **self._SAVE_LOAD_PARAMS)
            torch.save(obj = self.output_adapters,    f = model_path[2], **self._SAVE_LOAD_PARAMS)

    def load_model(self, model_path: Union[str, Tuple[str]]):
        if self._single_model:   
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            self.buildModel(torch.load(f=model_path, **self._SAVE_LOAD_PARAMS))
        else:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            self.input_adapters     = torch.load(f = model_path[0], **self._SAVE_LOAD_PARAMS)
            self.fno_and_proj_model = torch.load(f = model_path[1], **self._SAVE_LOAD_PARAMS)
            self.output_adapters    = torch.load(f = model_path[2], **self._SAVE_LOAD_PARAMS)
            self.buildModel((self.input_adapters, self.fno_and_proj_model, self.output_adapters))

    def load_data(self, file):
        pass

    def train(self, train_loader: Dataset, train_epochs: int):
        best_loss = np.inf        
        best_epoch = 0
        best_err = np.inf

        for epoch in range(train_epochs):
            train_err, avg_epoch_loss = self.train_single_epoch(epoch, train_loader, self._training_loss)
            print(f'{epoch} - th epoch: train error is {train_err}')

            if train_err < best_err:
                best_err = train_err

    @property
    def model(self):
        if self._single_model:
            return self.model
        else:
            return self.input_adapters, self.fno_and_proj_model, self.output_adapters

    def train_single_epoch(self, epoch, train_loader, training_loss):
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

        avg_epoch_loss = 0
        # avg_lasso_loss = 0
        
        if self._single_model:
            self.model.train()
        else:
            self.main_fno.train()

            for idx, _ in enumerate(self.input_adapters):
                self.input_adapters[idx].train()
                self.output_adapters[idx].train()


        if self.data_processor:
            self.data_processor.train()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = len(train_loader) # .size

        for idx, sample in enumerate(train_loader):
            
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_epoch_loss += loss.item()
                # if self.regularizer:
                #     avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        train_err /= len(train_loader)
        avg_epoch_loss /= self.n_samples
        # if self.regularizer:
        #     avg_lasso_loss /= self.n_samples
        # else:
        #     avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                avg_loss=avg_epoch_loss,
                train_err=train_err,
                # avg_lasso_loss=avg_lasso_loss,
                lr=lr
            )

        return train_err, avg_epoch_loss # , avg_lasso_loss

    def log_training(self, epoch, avg_loss, train_err, lr):
        self._logger.write('Epoch: {} | avg_loss: {} | train_err: {} | lr: {}'.format(epoch, avg_loss, train_err, lr))

    def on_epoch_start(self, *args, **kwargs):
        """
        Stub for implementing additional logick!
        """
        pass

    def train_one_batch(self, idx, sample, training_loss):
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

        self.optimizer.zero_grad(set_to_none=True)
        # if self.regularizer:
        #     self.regularizer.reset()
        # if self.data_processor is not None:
        #     sample = self.data_processor.preprocess(sample)
        # else:
        #     # load data to device if no preprocessor exists
            
        #     sample = {
        #         k: v.to(self.device)
        #         for k, v in sample.items()
        #         if torch.is_tensor(v)
        #     }

        if isinstance(Y, torch.Tensor):
            self.n_samples += Y.shape[0]
        else:
            self.n_samples += 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                if self._single_model:
                    out = self.model(X)
                else:
                    lift_out = self.input_adapters[eq_idx[0].item()](X)
                    main_out = self.main_fno(lift_out)
                    out = self.output_adapters[eq_idx[0].item()](main_out)

        else:
            if self._single_model:
                out = self.model(X)
            else:
                lift_out = self.input_adapters[eq_idx[0].item()](X)
                main_out = self.main_fno(lift_out)
                out = self.output_adapters[eq_idx[0].item()](main_out)
        
        # if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
        #     print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, Y = self.data_processor.postprocess(out, Y)

        loss = 0.0

        # print(f'Obtained out mean - {torch.mean(out)}, ref mean - {torch.mean(X)}')

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss += training_loss(out, X) # torch.nn.functional.mse_loss(out, X) #
        else:
            # loss += torch.nn.functional.mse_loss(out, Y) # training_loss(out, X)

            loss += training_loss(out, Y)
        
        # Heatmap(out[0, 0, ...].cpu().detach().numpy())
        
        return loss


    def finetune():
        pass