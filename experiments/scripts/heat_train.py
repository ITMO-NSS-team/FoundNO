import os
import argparse
from datetime import datetime

from typing import List

import glob
import sys

sys.path.append('.')

import torch

from neuralop.models import FNO

from fnofound.utils.training_utils import load_files_hdf5, validateOperator

from fnofound.utils.domains import Domain
from fnofound.utils.data_utils import SimpleDataset, NDDataset
from fnofound.utils.custom_trainer import Trainer, Logger

from fnofound.models.pecoda import PeCODANO
from fnofound.models.mamba_fno import PostLiftMambaFNO
from fnofound.models.localattn_exp import LocalAttnFNO

from fnofound.data import UnitGaussianNormalizer
from fnofound.data.data.transforms.data_processors import DefaultDataProcessor


# OPTIMIZER_PARAMS = {'optimizer': "adam", 'lr': 1e-3}

# SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 1e3}

def balanced_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, zero_threshold: float = 1e-6, eps: float = 1e-6):
    total_loss = 0.0
    C = pred.shape[1]
    for c in range(C):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        mask = torch.abs(t) > zero_threshold
        if mask.sum() == 0:
            continue
        diff_norm = torch.norm((p - t) * mask)
        target_norm = torch.norm(t * mask) + eps
        total_loss += diff_norm / target_norm

    return total_loss / C if C > 0 else torch.tensor(0.0, device=pred.device)

OPTIMIZER_PARAMS = {'optimizer': "adamw", 'lr': 1e-3, "weight_decay": 1e-5, "loss": balanced_rel_l2_loss}

SCHEDULER_PARAMS = {'scheduler': 'reducelr', 'patience': 15, 'factor': 0.5, 'min_lr': 1e-6}

ARGS = {'fno': {'model' : FNO,
                'params' : {'hidden_channels': 24,
                            'n_layers': 5,
                            'n_modes': [30, 30]}},
        'mambafno': {'model' : PostLiftMambaFNO,
                     'params' : {'modes': (64,64),
                                 'width': 32,
                                 'n_layers': 4,
                                 'use_mamba_kwargs': None,
                                 'mamba_fallback_kernel':9}},
        'localattnfno': {'model' : LocalAttnFNO,
                         'params' : {'width': 64,
                                     'n_local_layers': 2,
                                     'n_heads': 4, 
                                     'window_size': 127}},
        'pecoda': {'model' : PeCODANO,
                   'params' : {'hidden_variable_codimension': 16,
                               'n_layers': 2,
                               'n_layers_fno': 2,
                               'n_modes': [[64, 64], [64, 64], 64, 64]}}}

EXPNAME = 'heat'

if __name__ == "__main__":
    print(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'fno') # , type = ascii
    parser.add_argument("--epochs_max", default = 1e5, type = int)

    parser.add_argument("--single_model_location", default = '') # , type = ascii
    parser.add_argument("--lift_model_location",   default = '') # , type = ascii
    parser.add_argument("--main_model_location",   default = '') # , type = ascii
    parser.add_argument("--proj_model_location",   default = '') # , type = ascii

    args = parser.parse_args()

    data_dir = os.path.join(parent_dir, 'data', EXPNAME) #'../data/heat'
    filepaths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))

    print(f'Working with {EXPNAME} equation: parent_dir is {parent_dir}, data_dir is {data_dir}.')

    samples_pretrain = load_files_hdf5(filepaths, finetune = False, finetune_label = 'has_conv',
                                       to_standardize = True, original_shape = (51, 51))

    params = {'t': {'L': 1., 'n': samples_pretrain[0][0].shape[1]}, 'x': {'L': 1., 'n': samples_pretrain[0][0].shape[2]}}
    domain = Domain(params)

    # print('Samples shape: ', [(sample[0].shape, sample[1].shape) for sample in samples_pretrain])
    # raise NotImplementedError('!')
    
    batch_size = 10
    forcings  = torch.stack([sample[0] for sample in samples_pretrain])
    solutions = torch.stack([sample[1] for sample in samples_pretrain])
    solutions = solutions.permute(0, 1, 3, 2); forcings = forcings.permute(0, 1, 3, 2)
    print(f'forcings shape: {forcings.shape} & solutions shape: {solutions.shape} ')

    N = solutions.shape[0]
    perm = torch.randperm(N)
    solutions = solutions[perm]
    forcings  = forcings[perm]


    x = torch.linspace(0, 1, forcings.shape[-1])

    train_max_idx = int(solutions.shape[0] * 0.8)
    train_dataset = NDDataset(solutions[:train_max_idx], extra_channels = [forcings[:train_max_idx],], grids = [x,]) # XX, YY
    val_dataset   = NDDataset(solutions[train_max_idx:], extra_channels = [forcings[train_max_idx:],], grids = [x]) # XX, YY

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size)

    inp_normalizer = UnitGaussianNormalizer(dim = [2, 3])
    out_normalizer = UnitGaussianNormalizer(dim = [2, 3])

    for idx, sample in enumerate(train_dataset):
        sample_x = sample['x'].to('cuda')
        sample_y = sample['y'].to('cuda')

        if (idx % 100) == 0:
            print(f'Processing train sample {idx}: shapes are {sample_x.shape, sample_y.shape}, device: {sample_x.device}')
        inp_normalizer.partial_fit(sample_x)
        out_normalizer.partial_fit(sample_y)

    model_selection = ARGS[args.model]
    validateOperator(model_selection['model'], ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))

    model = model_selection['model'](in_channels = train_dataset.in_channels,
                                     out_channels = train_dataset.out_channels,
                                     **model_selection['params'])

    now = datetime.now()

    trainer = Trainer()
    logger_filename = os.path.join(parent_dir, 'experiments', 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.setLogger(filename = logger_filename)

    trainer.buildModel(model)
    trainer.buildOptimizer(n_dim = 2,
                            params_scheduler = SCHEDULER_PARAMS,
                            params_opt = OPTIMIZER_PARAMS)

    data_processor = DefaultDataProcessor(in_normalizer = inp_normalizer,
                                          out_normalizer = out_normalizer)

    trainer.to('cuda')
    trainer.train(train_loader, val_loader=val_loader, train_epochs=int(args.epochs_max), 
                  data_processor = [data_processor,])
    
    model = trainer.model

    model_savefile_base = os.path.join(parent_dir, 'experiments', 'pretrained_models')
    if trainer._single_model:
        if args.single_model_location == '':
            filename = f'{EXPNAME}_{args.model}_{now.day}_{now.hour}_{now.minute}.pt'
        else:
            filename = args.single_model_location

        model_savefile = os.path.join(model_savefile_base, filename)
    else:
        if args.lift_model_location == '':
            filename_lift = f'{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.pt'
        else:
            filename_lift = args.lift_model_location
        
        if args.main_model_location == '':
            filename_main = f'{EXPNAME}_{args.model}_main_{now.day}_{now.hour}_{now.minute}.pt'
        else:
            filename_main = args.main_model_location

        if args.proj_model_location == '':
            filename_proj = f'{EXPNAME}_{args.model}_proj_{now.day}_{now.hour}_{now.minute}.pt'
        else:
            filename_proj = args.proj_model_location


        model_savefile_lift = os.path.join(model_savefile_base, filename_lift)
        model_savefile_main = os.path.join(model_savefile_base, filename_main)
        model_savefile_proj = os.path.join(model_savefile_base, filename_proj)
        model_savefile = (model_savefile_lift, model_savefile_main, model_savefile_proj)

    trainer.saveModel(model_savefile)