import os
import warnings
import logging
import json
import csv
import time
from functools import reduce

from pathlib import Path

import math

from functools import singledispatchmethod
from typing import Tuple, List, Union

import numpy as np
import torch

import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

import dill

from neuralop.data.transforms.data_processors import DataProcessor

from .data_utils import Dataset, Heatmap
from torch.utils.data import DataLoader
from muno.models.muno import Muno
from muno.data.benchmarks.datasets import MultiPhysicsDataset

from .logger import Logger
from .optimizer_utils import set_optimizer, set_scheduler
from .training_utils import LpLoss


class Trainer(object):
    mixed_precision = False  # Load them from param json
    verbose = False
    eval_interval = 1000
    device = None

    _SAVE_LOAD_PARAMS = {}

    def __init__(self, backup_loc: str = None, device = None):# , devices: List[int] = [0,]):
        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        except KeyError:
            self.local_rank = 0

        try:
            self.global_rank = int(os.environ["RANK"])  
        except KeyError:
            self.local_rank = 0            

        if backup_loc is None:
            backup_loc = os.path.join(os.getcwd(), 'backup')
        self._backup_loc = backup_loc

        self.model = None
        self._best_val_loss = torch.inf

        self._history_csv_path = None
        self._history_jsonl_path = None
        self._training_start_time = None

        self.gradient_accumulation_steps = 1

        if device is not None:
            self.to(device)


    @singledispatchmethod
    def buildModel(self, model):
        raise NotImplementedError("Cannot declare model with anything, but nn Module or tuple of nn Modules")

    @buildModel.register
    def _(self, model: Muno):
        self._muno_model = True
        self.model = model
        self.params_to_optimize = [{'params': self.model.parameters()}, ]

        if self.device is not None:
            self.to(self.device)

    @buildModel.register
    def _(self, model: torch.nn.parallel.DistributedDataParallel):
        self._muno_model = True
        self.model = model
        self.params_to_optimize = [{'params': self.model.parameters()}, ]

        if self.device is not None:
            self.to(self.device)

    @buildModel.register
    def _(self, model: torch.nn.Module):
        self._muno_model = False
        self.model = model
        self.params_to_optimize = [{'params': self.model.parameters()}, ]

        if self.device is not None:
            self.to(self.device)

    @buildModel.register
    def _(self, model: tuple):  # expect Tuple[List[torch.nn.Module], torch.nn.Module, List[torch.nn.Module]]
        raise NotImplementedError('Depricated method, use Muno model instead.')
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

        # all changes in pdebench_multiphysics_pretrain.yaml ###########################################################
        if getattr(self, "train_main_fno", False):
            self.params_to_optimize.append({"params": self.main_fno.parameters()})

        for idx_expert_nn, _ in enumerate(self.input_adapters):
            self.params_to_optimize.append({"params": self.input_adapters[idx_expert_nn].parameters()})
            self.params_to_optimize.append({"params": self.output_adapters[idx_expert_nn].parameters()})

    def buildOptimizer(self,
                       n_dim: int,
                       params_scheduler: dict,
                       params_opt: dict,
                       trainer_loss=None):
        assert self.params_to_optimize is not None, 'Optimizer has to be constructed only after model declaration.'

        if self.model is None:
            raise RuntimeError("Model has not been declacred before optimizer initialization.")
        # if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
        #     self.optimizer = DistributedOptimizer()
            
        self.optimizer = set_optimizer(params_opt, self.params_to_optimize)
        self.scheduler = set_scheduler(params_scheduler, self.optimizer)

        if trainer_loss is None:
            self._training_loss = LpLoss(d=n_dim)
        else:
            self._training_loss = trainer_loss

    def to(self, device: str = 'cuda'):
        self.device = device

        # if self._single_model:
        self.model.to(device)

        # else:
        #     if self.main_fno is None or self.input_adapters is None or self.output_adapters is None:
        #         raise AttributeError('Hidden Fourier NO layers and projection or liftings are not yet declared.')

        #     self.main_fno.to(device)
        #     for idx, _ in enumerate(self.input_adapters):
        #         self.input_adapters[idx].to(device)
        #         self.output_adapters[idx].to(device)

    def setLogger(self, filename, logger: Logger = None, log_level=logging.INFO, logger_name: str = "FoundationalFNO"):
        if logger is None:
            self._logger = Logger(
                filename=filename,
                log_level=log_level,
                logger_name=logger_name,
                write_every=1,
                epochs_aggreg=1,
                info_entries=["train_loss", "val_loss", "lr", "local_rank", "global_rank"],
            )
        else:
            self._logger = logger

    def setHistoryLogger(self, log_dir):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._history_csv_path = log_dir / "history.csv"
        self._history_jsonl_path = log_dir / "history.jsonl"

        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "lr",
            "epoch_seconds",
            "elapsed_seconds",
            "best_val_loss",
            "checkpoint_saved",
        ]

        with open(self._history_csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

        with open(self._history_jsonl_path, "w", encoding="utf-8"):
            pass

    def writeEpochHistory(
            self,
            epoch,
            train_loss,
            val_loss,
            lr,
            epoch_seconds,
            checkpoint_saved,
    ):
        if self._history_csv_path is None or self._history_jsonl_path is None:
            return

        elapsed_seconds = time.perf_counter() - self._training_start_time

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr) if lr is not None else None,
            "epoch_seconds": float(epoch_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "best_val_loss": float(self._best_val_loss),
            "checkpoint_saved": bool(checkpoint_saved),
        }

        with open(self._history_csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=record.keys())
            writer.writerow(record)

        with open(self._history_jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def saveModel(self, model_path: Tuple[str, List[str]]): # Union[str, ] 
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = self.model.module 
        else:
            model = self.model

        if self._muno_model:
            assert isinstance(model_path, tuple) and len(model_path) == 3, \
                'Saving lifting-main part-projection model requires tuple of str arg with len 3'
            assert isinstance(model, Muno), \
                f'Somehow, using non-Muno model in the Muno case, instead got {type(model)}.'
            model.save(model_path)

        else:
            assert isinstance(model_path, str), 'Saving of a single model requires a single path str argument'
            torch.save(obj=model, f=model_path, pickle_module=dill, **self._SAVE_LOAD_PARAMS)


    def loadModel(self, model_path: Union[str, Tuple[None, str, Tuple[str]]], # use_data_parallel: bool = False, devices: Union[int, List[int]] = [], 
                  **_SAVE_LOAD_PARAMS):
        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, pickle_module=dill, **_SAVE_LOAD_PARAMS)
        else:
            self.model = Muno.load(f=model_path, **_SAVE_LOAD_PARAMS)
            # if use_data_parallel:
            #     self.model = self.model.toDataParallel(devices=devices)

        # model = basicLoadModel(model_path, self._SAVE_LOAD_PARAMS)
        # if isinstance(model, tuple):
        #     self._single_model = False

        #     self.input_adapters = model[0]
        #     self.main_fno = model[1]
        #     self.output_adapters = model[2]
        # else:
        #     self.model = model
        #     self.params_to_optimize = [{'params': self.model.parameters()}, ]
        #     self._single_model = True

    def loadData(self, file):
        pass

    def train(self, train_loader: MultiPhysicsDataset, val_loader: MultiPhysicsDataset,
              train_epochs: int, data_processor: Union[list, DataProcessor] = None, GA_size: int = 4):
        '''
        refactored from train_loader: Union[DataLoader, list], val_loader: Union[DataLoader, list] to 
        train_loader: muno.data.benchmarks.datasets.MultiPhysicsDataset, 
        val_loader: muno.data.benchmarks.datasets.MultiPhysicsDataset,
        '''
        # if isinstance(train_loader, DataLoader):
        #     train_loader = [train_loader, ]
        # if isinstance(val_loader, DataLoader):
        #     val_loader = [val_loader, ]

        # track number of training examples in batch
        self.n_samples = len(train_loader)
        self.n_samples_val = len(val_loader)

        # if isinstance(data_processor, DataProcessor):
        #     data_processor = [data_processor, ]
        # elif data_processor is None:
        #     data_processor = [None, ]

        self._training_start_time = time.perf_counter()

        log_dir = Path(self._logger._filename).parent
        self.setHistoryLogger(log_dir)

        best_train_loss = np.inf

        # if self._single_model:
        n_params = sum(p.numel() for p in self.model.parameters())
        init_log = 'Initializing training of model of type' + \
                    ' {} | epochs: {} | n params: {}'.format(type(self.model),
                                                            train_epochs,
                                                            n_params)

        # else:
        #     n_params = sum(p.numel() for p in self.input_adapters[0].parameters()) + \
        #                sum(p.numel() for p in self.main_fno.parameters()) + \
        #                sum(p.numel() for p in self.output_adapters[0].parameters())

        #     init_log = 'Initializing training of model of type' + \
        #                ' {}, {}, {} | epochs: {} | n params: {}'.format(type(self.input_adapters[0]),
        #                                                                 type(self.main_fno),
        #                                                                 type(self.output_adapters[0]),
        #                                                                 train_epochs,
        #                                                                 n_params)

        self._logger.write(init_log)

        for epoch in range(train_epochs):
            train_loss, val_loss = self.trainSingleEpoch(
                epoch,
                train_loader,
                val_loader,
                self._training_loss,
                data_processor,
            )

            # print(f"Epoch {epoch} on globrank {self.global_rank} localrank {self.local_rank}: train_loss={train_loss}, val_loss={val_loss}")

            if train_loss < best_train_loss:
                best_train_loss = train_loss

            if epoch == train_epochs:
                self.logTraining(train_loss=train_loss, val_loss=val_loss, lr=0)

        return self.model
    
        # if self._single_model:
        #     return self.model
        # else:
        #     return self.input_adapters, self.main_fno, self.output_adapters

    def trainSingleEpoch(self, epoch, train_loader: List[DataLoader], val_loader: List[DataLoader],
                         training_loss, data_processor: None, # List[DataProcessor] = [None, ],  # training: bool = True,
                         GA_size: int = 4):
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
        epoch_start_time = time.perf_counter()

        self.onEpochStart(epoch)

        self.model.train()

        if data_processor is not None: # [0]
            # for idx in range(len(data_processor)):
            data_processor.train()

        train_loss = 0.0
        n_fine_samples = self.n_samples
        accumulation_steps = int(self.gradient_accumulation_steps)
        accumulated_loss = torch.tensor([0.0, ], dtype=torch.float32, requires_grad=True, device=self.device)

        for idx, sample in enumerate(train_loader):
            loss = self.trainOneBatch(
                epoch,
                sample,
                training_loss,
                data_processor,
                training=True,
            )

            if torch.isnan(loss).item():
                print("loss is NaN")
                n_fine_samples -= 1
                continue

            accumulated_loss = accumulated_loss + loss #.item()
            with torch.no_grad():
                train_loss += loss.item()

            if (idx + 1) % accumulation_steps == 0:
                accumulated_loss.backward()
                self.optimizer.step()
                accumulated_loss = torch.tensor([0.0, ], dtype=torch.float32, requires_grad=True, device=self.device)
                self.optimizer.zero_grad(set_to_none=True)

        if not torch.isclose(accumulated_loss, torch.tensor([0.0, ], dtype=torch.float32, device=self.device)):
            accumulated_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)                

        del loss, accumulated_loss
        torch.cuda.empty_cache()

        train_loss /= n_fine_samples

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_loss)
        else:
            self.scheduler.step()

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]

        self.model.eval()

        with torch.no_grad():
            val_loss = 0.

            n_fine_samples = self.n_samples_val

            for idx, sample in enumerate(val_loader):
                loss = self.trainOneBatch(epoch, sample, training_loss,
                                          data_processor, training=False)
                if torch.isnan(loss).item():
                    n_fine_samples -= 1
                    continue

                val_loss += loss.item()

            val_loss /= n_fine_samples

        self.logTraining(val_loss=val_loss, train_loss=train_loss, lr=lr,
                         local_rank=self.local_rank, global_rank=self.global_rank)

        checkpoint_saved = self.onEpochEnd(val_loss=val_loss)
        epoch_seconds = time.perf_counter() - epoch_start_time

        self.writeEpochHistory(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            epoch_seconds=epoch_seconds,
            checkpoint_saved=checkpoint_saved,
        )

        return train_loss, val_loss

    def logTraining(self, val_loss, train_loss, lr, local_rank: int = 0, global_rank: int = 0):
        self._logger.write(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "local_rank": local_rank,
                "global_rank": global_rank
            }
        )

    def onEpochStart(self, *args, **kwargs):
        """
        Stub for implementing additional logick!
        """
        pass

    @property
    def save_paths(self):
        return self._save_paths

    @save_paths.setter
    def save_paths(self, paths):
        if self._muno_model:
            assert isinstance(paths, (tuple, list)), \
                "Save paths have to be tuple/list in case of a multiple adapter model."
            
            assert isinstance(paths[1], str), \
                "Save path for a core has to be a string in case of a multiple adapter model."

            assert all(isinstance(paths[idx], (tuple, list)) for idx in [0, 2]), \
                "Save paths for adapters have to be passed as list/tuple of strings."

        else:
            assert isinstance(paths, str), \
                "Save paths have to be strings in case of a single model."
        self._save_paths = paths

    def onEpochEnd(self, *args, **kwargs):
        """
        Save model checkpoint when validation loss improves.
        Returns True if a checkpoint was saved on this epoch.
        """
        checkpoint_saved = False

        if kwargs["val_loss"] < self._best_val_loss:
            self._best_val_loss = kwargs["val_loss"]
            checkpoint_saved = True

            directory = Path(self._backup_loc)
            directory.mkdir(parents=True, exist_ok=True)

            # if self._muno_model:
            #     assert isinstance(self._save_paths, (tuple, list)), (
            #         "Save paths have to be tuple/list in case of a multiple adapter model."
            #     )
            #     assert isinstance(self._save_paths[1], str), (
            #         "Save path for a core has to be a string in case of a multiple adapter model."
            #     )
            #     assert all(
            #         isinstance(self._save_paths[idx], (tuple, list))
            #         for idx in [0, 2]
            #     ), "Save paths for adapters have to be passed as list/tuple of strings."

            #     self.saveModel(self._save_paths)
            # else:
            #     assert isinstance(self._save_paths, str), (
            #         "Save paths have to be strings in case of a single model."
            #     )
            self.saveModel(self.save_paths)                

        return checkpoint_saved

    def trainOneBatch(self, idx, sample, training_loss, data_processor=None, training: bool = False):
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
        assert isinstance(sample, dict), 'Sample has to be passed as dict.'
        test_key = list(sample.keys())[0]
        assert isinstance(sample[test_key], dict), \
            'A sample, obtained for a single-physics dataset has to be a dict.'

        for key in sample.keys():    
            sample[key]["x"] = sample[key]["x"].to(self.device)
            sample[key]["y"] = sample[key]["y"].to(self.device)

            if "mask" in sample[key].keys():
                sample[key]["mask"] = sample[key]["mask"].to(self.device)

        if data_processor is not None:
            if isinstance(sample, dict):
                sample = data_processor.preprocess(sample, training=training)
            else:
                warnings.warn('Possibly, incorrect type of model input')

        if self.mixed_precision:
            raise NotImplementedError('No mixed precision functionality implemented!')
            with torch.autocast(device_type=self.autocast_device_type):
                if self._single_model:
                    out = self.model(sample["x"])
                else:
                    out = self.input_adapters[sample["eq_idx"][0].item()](sample["x"])
                    out = self.main_fno(out)
                    out = self.output_adapters[sample["eq_idx"][0].item()](out)

        else:
            out = self.model({key: sample[key]["x"] for key in sample})

        if data_processor is not None:
            out, sample = data_processor.postprocess(out, sample, training=training)

        mask = {bkey: (sample[bkey]["mask"] if "mask" in sample[bkey] else None) for bkey in sample.keys()}

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                if isinstance(training_loss, list):
                    loss = reduce(torch.add, [reduce(torch.add, 
                                                     [loss_func(out[bkey], sample[bkey]["y"], mask[bkey]) for loss_func in training_loss]) 
                                  for bkey in sample.keys()])
                else:
                    loss = reduce(torch.add, [training_loss(out[bkey], sample[bkey]["y"], mask[bkey]) for bkey in sample.keys()]) 
        else:
            if isinstance(training_loss, list):
                loss = reduce(torch.add, [reduce(torch.add,
                                                 [loss_func(out[bkey], sample[bkey]["y"], mask[bkey]) for loss_func in training_loss]) 
                              for bkey in sample.keys()])
            else:
                loss = reduce(torch.add, [training_loss(out[bkey], sample[bkey]["y"], mask[bkey]) for bkey in sample.keys()]) 

        return loss

    def finetune(self):
        pass