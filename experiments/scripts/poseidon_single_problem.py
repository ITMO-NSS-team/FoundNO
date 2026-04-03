import os
import argparse
from datetime import datetime

from typing import List, Tuple

import glob
import sys

sys.path.append('.')

import torch

from neuralop.models import FNO
# from fnofound.models.fno import FNO

from fnofound.utils.training_utils import load_files_hdf5, validateOperator

from fnofound.utils.domains import Domain
from fnofound.utils.data_utils import SimpleDataset, NDDataset
from fnofound.utils.custom_trainer import Trainer, Logger

from fnofound.models.pecoda import PeCODANO
from fnofound.models.mamba_fno import PostLiftMambaFNO3D, PostLiftMambaLifting

from fnofound.models.localattn_exp import LocalAttnFNO

from fnofound.data import UnitGaussianNormalizer
from fnofound.data.data.transforms.data_processors import DefaultDataProcessor

from neuralop.layers.channel_mlp import ChannelMLP

import xarray as xr
from torch.nn import MSELoss


def balanced_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, zero_threshold: float = 1e-6, eps: float = 1e-6):
    total_loss = 0.0
    C = pred.shape[1]
    for c in range(C):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        # mask = torch.abs(t) > zero_threshold
        # if mask.sum() == 0:
        #     continue
        diff_norm = torch.norm((p - t)) #  * mask
        target_norm = torch.norm(t) + eps #  * mask
        total_loss += diff_norm / target_norm

    return total_loss / C if C > 0 else torch.tensor(0.0, device=pred.device)


OPTIMIZER_PARAMS = {'optimizer': "adamw", 'lr': 1e-4, "weight_decay": 1e-5, "loss": MSELoss} #balanced_rel_l2_loss} adamw

SCHEDULER_PARAMS = {'scheduler': 'reducelr', 'patience': 4, 'factor': 0.5, 'min_lr': 1e-6}

ARGS = {'fno': {'model' : FNO,
                'params' : {'hidden_channels': 44,
                            'n_layers': 5,
                            'n_modes': [6, 40, 40]}},
        'mambafno': {'model' : PostLiftMambaFNO3D,
                     'params' : {'modes': (6, 60, 60),
                                 'width': 44,
                                 'n_layers': 5,
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
                               'n_modes': [[64, 64], [64, 64], 64, 64]}},
        'adapted_fno': {'model': [PostLiftMambaLifting, FNO, ChannelMLP],
                       'params': [{'width': 20,
                                   'use_mamba_kwargs': None,
                                   'mamba_fallback_kernel': 9,
                                   'padding': 0,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu},
                                  {'hidden_channels': 20,
                                   'n_layers': 4,
                                   'n_modes': [10, 40, 40],  # [8, 32, 32]
                                   'disable_lifting_and_projection': True 
                                   },
                                  {'hidden_channels': 20,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu}]},
        'adapted_fno_no_mamba': {'model': [ChannelMLP, FNO, ChannelMLP],
                       'params': [{'hidden_channels': 32,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu},
                                  {'hidden_channels': 32,
                                   'n_layers': 4,
                                   'n_modes': [20, 42, 42],  # [8, 32, 32]
                                   'disable_lifting_and_projection': True 
                                   },
                                  {'hidden_channels': 32,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu}]}}

EXPNAME = 'flows'


def loadNcdfData(filename: str, dtype) -> Tuple[int, torch.Tensor]:
    with xr.open_dataset(filename) as dataSet: # '/media/mikemaslyaev/Data/Poseidon_data/CE_GAUSS/data_0.nc'
        try:
            data = torch.from_numpy(dataSet['data'].to_numpy()).to(dtype)
        except KeyError:
            data = torch.from_numpy(dataSet['velocity'].to_numpy()).to(dtype)
            
    data = data.swapaxes(1, 2)
    return data.shape[1], data

def getLoaderChannels(dataloader) -> Tuple[int, int]:
    for batch in dataloader:
        in_channels = batch['x'].shape[1]
        out_channels = batch['y'].shape[1]

        break

    return in_channels, out_channels


if __name__ == "__main__":
    print(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'fno') # , type = ascii
    parser.add_argument("--epochs_max", default = 1e5, type = int)

    parser.add_argument("--data_location", default='')

    parser.add_argument("--single_model_location", default = '') # , type = ascii
    parser.add_argument("--lift_model_location",   default = '') # , type = ascii
    parser.add_argument("--main_model_location",   default = '') # , type = ascii
    parser.add_argument("--proj_model_location",   default = '') # , type = ascii

    args = parser.parse_args()
    
    # data_dir = '/media/mikemaslyaev/Data/Poseidon_data/CombinedDatasets'
    # filepaths = sorted(glob.glob(os.path.join(data_dir, '*.nc')))

    # if len(args.data_location):
    filepaths = ['/media/mikemaslyaev/Data/Poseidon_data/NS_SINES/velocity_0.nc',
                '/media/mikemaslyaev/Data/Poseidon_data/NS_GAUSS/velocity_2.nc',
                '/media/mikemaslyaev/Data/Poseidon_data/NS_GAUSS/velocity_3.nc',
                '/media/mikemaslyaev/Data/Poseidon_data/NS_GAUSS/velocity_4.nc',
                '/media/mikemaslyaev/Data/Poseidon_data/NS_GAUSS/velocity_5.nc',
                #  '/media/mikemaslyaev/Data/Poseidon_data/NS_GAUSS/velocity_6.nc',
                '/media/mikemaslyaev/Data/Poseidon_data/NS_SINES/velocity_7.nc',]

    print(f'Loading data from filepaths: {filepaths}')

    params = {'t': {'L': 1., 'n': 128}, 'x': {'L': 1., 'n': 128}}
    domain = Domain(params)

    train_dataloaders = []
    val_loaders       = []
    data_processors   = []

    for fidx, filepath in enumerate(filepaths):
        print(f'Loading dataset from {filepath}')
        sample_max = 500

        channels, data = loadNcdfData(filepath, dtype = torch.float32)
        data = data[:sample_max]

        if channels == 3:
            cur_forcings = data[:, 2:]
            cur_solutions = data[:, :2]
        if channels == 5:
            cur_forcings = data[:, (0, 3, 4)]
            cur_solutions = data[:, (1, 2)]

        del data

        if fidx == 0:
            solutions = cur_solutions
            forcings = cur_forcings
        else:
            solutions = torch.cat([solutions, cur_solutions,], dim = 0)
            forcings = torch.cat([forcings, cur_forcings,], dim = 0)
        print('Loaded!')

    print(f'Shape of forcings {forcings.shape} and solutions {solutions.shape}')
    batch_size = 1

    N = solutions.shape[0]
    perm = torch.randperm(N)
    solutions = solutions[perm, ...]
    forcings = forcings[perm, ...]

    inp_normalizer = UnitGaussianNormalizer(dim = [2, 3, 4]) # 
    out_normalizer = UnitGaussianNormalizer(dim = [2, 3, 4])#, mask = mask)

    H, W = 128, 128
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    X_grid, Y_grid = torch.meshgrid(y, x, indexing='ij')
    T = forcings.shape[1]
    t_grid = torch.linspace(0, 1, T)

    train_max_idx = int(solutions.shape[0] * 0.8)
    train_dataset = NDDataset(solutions[:train_max_idx], extra_channels = [forcings[:train_max_idx],], 
                              grids = None, dataset_index=0) # [X_grid, Y_grid] XX, YY
    val_dataset   = NDDataset(solutions[train_max_idx:], extra_channels = [forcings[train_max_idx:],],
                              grids = None, dataset_index=0) # [X_grid, Y_grid] XX, YY

    for idx, sample in enumerate(train_dataset):
        sample_x = sample['x'].to('cuda')
        sample_y = sample['y'].to('cuda')

        if (idx % 100) == 0:
            print(f'Processing train sample {idx}: shapes are {sample_x.shape, sample_y.shape}, device: {sample_x.device}')
        inp_normalizer.partial_fit(sample_x)
        out_normalizer.partial_fit(sample_y)
    print('inp_normalizer.mean()', inp_normalizer.mean, 'inp_normalizer.std()', inp_normalizer.std) 
    print('out_normalizer.mean()', out_normalizer.mean, 'out_normalizer.std()', out_normalizer.std) 


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = batch_size)
    train_dataloaders.append(train_loader)
    val_loaders.append(val_loader)

    data_processor = DefaultDataProcessor(in_normalizer = inp_normalizer,
                                            out_normalizer = out_normalizer)
    data_processors.append(data_processor)

    model_selection = ARGS[args.model]

    if isinstance(model_selection['model'], (tuple, list)):
        model = list()
        for idx, submodel in enumerate(model_selection['model']):
            if idx == 0:
                liftings = []
                for data_idx, loader in enumerate(train_dataloaders):
                    in_channels, _ = getLoaderChannels(loader)
                    # in_channels = set.in_channels
                    # key =
                    out_channels = model_selection['params'][idx]['hidden_channels']
                    liftings.append(submodel(in_channels=in_channels,
                                             out_channels=out_channels,
                                             **model_selection['params'][idx]))
                model.append(liftings)
    
            elif idx == len(model_selection['model']) - 1:
                projections = []
                for data_idx, set in enumerate(train_dataloaders):
                    in_channels = model_selection['params'][idx]['hidden_channels']
                    _, out_channels = getLoaderChannels(loader)
                    projections.append(submodel(in_channels=in_channels,
                                                out_channels=out_channels,
                                                **model_selection['params'][idx]))
                model.append(projections)
    
            else:
                in_channels = model_selection['params'][idx]['hidden_channels']
                out_channels = model_selection['params'][idx]['hidden_channels']
                model.append(submodel(in_channels=in_channels,
                                      out_channels=out_channels,
                                      **model_selection['params'][idx]))
    
        assert len(model) == 3, 'Something went wrong!'
        model = tuple([model[0], model[1], model[2]])
    else:
        assert len(train_dataloaders) == 1, 'Trying to train a single lift-proj model on multiple datasets'
        validateOperator(model_selection['model'],
                          ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))

        in_channels, out_channels = getLoaderChannels(train_dataloaders[0])
        print(f'dataset channels: in - {in_channels}, out - {out_channels}')
        model = model_selection['model'](in_channels  = in_channels,
                                         out_channels = out_channels,
                                         **model_selection['params'])
        
    # print('len(model[0])', len(model[0]), 'len(model[2])', len(model[2]))
    now = datetime.now()

    trainer = Trainer()
    logger_filename = os.path.join(parent_dir, 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.setLogger(filename = logger_filename)

    trainer.buildModel(model)
    if SCHEDULER_PARAMS['scheduler'] == 'cosine':
        SCHEDULER_PARAMS['max_cosine_lr_epochs'] = args.epochs_max
    trainer.buildOptimizer(n_dim = 3,
                           params_scheduler = SCHEDULER_PARAMS,
                           params_opt = OPTIMIZER_PARAMS)

    trainer.to('cuda')
    trainer.train(train_loader=train_dataloaders, val_loader=val_loaders, train_epochs=int(args.epochs_max), 
                  data_processor = data_processors)
    
    model_savefile_base = os.path.join(parent_dir, 'pretrained_models')
    if trainer._single_model:
        if args.single_model_location == '':
            filename = f'{EXPNAME}_{args.model}_{now.day}_{now.hour}_{now.minute}.pt'
        else:
            filename = args.single_model_location

        model_savefile = os.path.join(model_savefile_base, filename)
    else:
        if args.lift_model_location == '':
            
            filename_lift = []
            for idx in range(len(model[0])):
                filename_lift.append(os.path.join(model_savefile_base, 
                                                  f'{EXPNAME}_{idx}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.pt'))
        else:
            filename_lift = args.lift_model_location
        
        if args.main_model_location == '':
            filename_main = os.path.join(model_savefile_base, f'{EXPNAME}_{args.model}_main_{now.day}_{now.hour}_{now.minute}.pt')
        else:
            filename_main = args.main_model_location

        if args.proj_model_location == '':
            filename_proj = []
            for idx in range(len(model[2])):
                filename_proj.append(os.path.join(model_savefile_base, 
                                                  f'{EXPNAME}_{idx}_{args.model}_proj_{now.day}_{now.hour}_{now.minute}.pt'))
        else:
            filename_proj = args.proj_model_location


        # model_savefile_lift = os.path.join(model_savefile_base, filename_lift)
        # model_savefile_main = os.path.join(model_savefile_base, filename_main)
        # model_savefile_proj = os.path.join(model_savefile_base, filename_proj)
        model_savefile = (filename_lift, filename_main, filename_proj)

    trainer.saveModel(model_savefile)
    for idx, processor in enumerate(data_processors):
        processor.in_normalizer.to_file(f'inp_norm_{idx}.pkl')
        processor.out_normalizer.to_file(f'out_norm_{idx}.pkl')

    # out_normalizer.to_file(f'out_norm_{args.var_key}.pkl')