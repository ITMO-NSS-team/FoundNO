import os
import gc
import argparse
import h5py
from datetime import datetime
import numpy as np

from typing import List, Tuple

import time
import sys
from pathlib import Path

sys.path.append('.')

import torch
import torch.multiprocessing as mp

from neuralop.models import UNO, FNO
# from muno.models.fno import FNO

from muno.utils.training_utils import load_files_hdf5, validateOperator

from muno.utils.domains import Domain
from muno.utils.data_utils import SimpleDataset, NDDataset, syncSuffle
from muno.utils.custom_trainer import Trainer, Logger
from muno.utils.training_utils import BalancedRelL2Loss, FourierHFLoss

# from muno.models.pecoda import PeCODANO
from muno.models.mamba_fno import PostLiftMambaFNO, PostLiftMambaLifting, PostLiftMambaUNO

from muno.models.localattn_exp import LocalAttnFNO

from muno.data import UnitGaussianNormalizer
from muno.data.data.transforms.data_processors import DefaultDataProcessor

from neuralop.layers.channel_mlp import ChannelMLP

import xarray as xr

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


OPTIMIZER_PARAMS = {'optimizer': "adamw", 'lr': 1e-3, "weight_decay": 1e-2} #balanced_rel_l2_loss} adamw

#OPTIMIZER_PARAMS = {'optimizer': 'lbfgs', 'lr': 1e-1}
# SCHEDULER_PARAMS = {'scheduler': 'reducelr', 'patience': 8, 'factor': 0.5, 'min_lr': 1e-6}

SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 5e2}

ARGS = {'fno': {'model' : UNO,
                'params' : {'hidden_channels': 80,
                            'n_layers': 5,
                            'uno_n_modes': [[16, 30, 30],]*5,
                            'uno_out_channels': [80, 80, 80, 80, 80], # 30
                            'uno_scalings': [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [1., 1., 1.]], # [1.0,1.0,1.0], 
                            'non_linearity': torch.nn.functional.gelu,
                            'horizontal_skips_map':{4:0, 3:1}, # 3:1
			                'channel_mlp_skip': "linear",
                            'channel_mlp_dropout': 0.2,
                            'factorization': 'tucker',
                            'rank': 0.05,
                            'implementation': 'factorized'}},
        'mambauno': {'model' : PostLiftMambaUNO,
                     'params' : {'uno_n_modes': [[16, 50, 50],] * 4,
                                 'width': 30,
                                 'uno_out_channels': [30, 30, 30, 30],
                                 'uno_scalings': [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [1.,1.,1.]],
                                 'n_layers': 4,
                                 'horizontal_skips_map': {3:0,},
                                 'use_mamba_kwargs': None,
                                 'mamba_fallback_kernel':9}},
        'mambafno': {'model' : PostLiftMambaFNO,
                     'params' : {'modes': (20, 40, 40),
                                 'width': 65,
                                 'n_layers': 4,
                                 'use_mamba_kwargs': None,
                                 'mamba_fallback_kernel':9}},
        'localattnfno': {'model' : LocalAttnFNO,
                         'params' : {'width': 64,
                                     'n_local_layers': 2,
                                     'n_heads': 4,
                                     'window_size': 127}},
        #'pecoda': {'model' : PeCODANO,
        #           'params' : {'hidden_variable_codimension': 16,
        #                       'n_layers': 2,
        #                       'n_layers_fno': 2,
        #                       'n_modes': [[64, 64], [64, 64], 64, 64]}},
        'adapted_fno': {'model': [PostLiftMambaLifting, UNO, ChannelMLP],
                       'params': [{'width': 80,
                                   'use_mamba_kwargs': None,
                                   'mamba_fallback_kernel': 9,
                                   'padding': 0,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu},
                                  {'hidden_channels': 80,
                                    'n_layers': 5,
                                    'uno_n_modes': [[14, 32, 32],]*5,
                                    'uno_out_channels': [80, 80, 80, 80, 80], #  30,
                                    'uno_scalings': [[1.,1.,1.],]*5, # [1.,1.,1.], [1.,1.,1.], [1.,1.,1.]], # [0.5,0.5,0.5], 
                                    'non_linearity': torch.nn.functional.gelu,
                                    'horizontal_skips_map':{4:0, 3:1}, # 4:0, 
                                    'channel_mlp_skip': "linear"},
                                  {'hidden_channels': 80,
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

EXPNAME = 'shear_flows'

# def loadNcdfData(filename: str, dtype) -> Tuple[int, torch.Tensor]:
#     with xr.open_dataset(filename, engine="h5netcdf") as dataSet: # '/media/mikemaslyaev/Data/Poseidon_data/CE_GAUSS/data_0.nc'
#         try:
#             data = torch.from_numpy(dataSet['data'].to_numpy()).to(dtype)
#         except KeyError:
#             data = torch.from_numpy(dataSet['velocity'].to_numpy()).to(dtype)
            
#     data = data.swapaxes(1, 2)
#     return data.shape[1], data

def loadShearFlowData(filename: str, dtype: torch.dtype = torch.float32) -> Tuple[int, torch.Tensor]:
    with h5py.File(filename, 'r') as h5_file:
        v = torch.from_numpy(h5_file['t1_fields']['velocity'][...]).permute(0, 4, 1, 2, 3).to(dtype)  # velocities

        p = torch.from_numpy(h5_file['t0_fields']['pressure'][...]).unsqueeze(1).to(dtype)            # pressure
        v = torch.concat((v, p), dim = 1)
        del p
        print("First concat executed")

        t = torch.from_numpy(h5_file['t0_fields']['tracer'][...]).unsqueeze(1).to(dtype)              # tracer
        v = torch.concat((v, p), dim = 1)
        del t

        print(v.shape, p.shape, t.shape)

        # print(f['dimensions']['time'][0], f['dimensions']['time'][-1], f['dimensions']['time'].size)
        T = (h5_file['dimensions']['time'][0], h5_file['dimensions']['time'][-1], h5_file['dimensions']['time'].size)
        X = (h5_file['dimensions']['x'][0], h5_file['dimensions']['x'][-1], h5_file['dimensions']['x'].size)
        Y = (h5_file['dimensions']['y'][0], h5_file['dimensions']['y'][-1], h5_file['dimensions']['y'].size)
        
        # Re = torch.tensor(h5_file['scalars']['Reynolds'][...]).expand(p.shape)
        # Sc = torch.tensor(h5_file['scalars']['Schmidt'][...]).expand(p.shape)

        # print('Before concat')

        # with torch.no_grad():
        time.sleep(20)
        # v = torch.concat((v, p, t), dim = 1) # , Re, Sc
    
        return v.shape[1], (T, X, Y), v # (Re, Sc), 

    # return data.shape[1], data


def getLoaderChannels(dataloader) -> Tuple[int, int]:
    for batch in dataloader:
        in_channels = batch['x'].shape[1]
        out_channels = batch['y'].shape[1]

        break

    return in_channels, out_channels


if __name__ == "__main__":
    # mp.set_start_method('spawn')

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
    

    filepaths = ['/media/mikemaslyaev/Data/the_well/shear_flow/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5',]
                #  '/media/mikemaslyaev/Data/the_well/shear_flow/shear_flow_Reynolds_1e4_Schmidt_1e1.hdf5',]    #'/media/mikemaslyaev/Data/Poseidon_data/NS_SINES/velocity_0.nc',]

    print(f'Loading data from filepaths: {filepaths}')

    params = {'t': {'L': 1., 'n': 128}, 'x': {'L': 1., 'n': 128}}
    domain = None # Domain(params)

    train_dataloaders = []
    val_loaders       = []
    data_processors   = []

    for fidx, filepath in enumerate(filepaths):
        print(f'Loading dataset from {filepath}')
        sample_max = 8000 # -1

        channels, (T, X, Y), data = loadShearFlowData(filepath, dtype = torch.float32)
        if domain is None:
            params = {'t': {'L': T[1] - T[0], 'n': T[2]}, 'x': {'L': X[1] - X[0], 'n': X[2]}, 'y': {'L': Y[1] - Y[0], 'n': Y[2]}}
            domain = Domain(params)

#        print('Ref count after loading: ', sys.getrefcount(data))
        data = data[:sample_max]

        # if channels == 3:
        #     cur_forcings = data[:, 2:]
        #     cur_solutions = data[:, :2]
        # if channels == 5:
        #     cur_forcings = data[:, (0, 3, 4)]
        #     cur_solutions = data[:, (1, 2)]

        cur_forcings = data[:, 3:]
        cur_solutions = data[:, :3]

        del data

        cur_solutions, cur_forcings = syncSuffle(cur_solutions, cur_forcings)

        train_max_idx = int(cur_solutions.shape[0] * 0.8)

        if fidx == 0:
            solutions_train = cur_solutions[:train_max_idx] # .swapaxes(-1, -2)
            forcings_train  = cur_forcings[:train_max_idx]  # .swapaxes(-1, -2)

            solutions_test  = cur_solutions[train_max_idx:] # .swapaxes(-1, -2)
            forcings_test   = cur_forcings[train_max_idx:]  # .swapaxes(-1, -2)

#            print('Ref count after init: ', sys.getrefcount(solutions_train))
        else:
            solutions_train = torch.cat([solutions_train, cur_solutions[:train_max_idx],], dim = 0) # .swapaxes(-1, -2)
            forcings_train = torch.cat([forcings_train, cur_forcings[:train_max_idx],], dim = 0) # .swapaxes(-1, -2)

            solutions_test  = torch.cat([solutions_test, cur_solutions[train_max_idx:],], dim = 0) # .swapaxes(-1, -2)
            forcings_test  = torch.cat([forcings_test, cur_forcings[train_max_idx:],], dim = 0) # .swapaxes(-1, -2)
        print('Loaded!')

    print(f'Shape of forcings {solutions_train.shape} & {solutions_test.shape} and \
            solutions {solutions_train.shape} & {solutions_test.shape}')
    batch_size = 1

    inp_normalizer = UnitGaussianNormalizer(dim = [2, 3, 4])
    out_normalizer = UnitGaussianNormalizer(dim = [2, 3, 4])

    H, W = 128, 128
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    X_grid, Y_grid = torch.meshgrid(y, x, indexing='ij')
    T = forcings_train.shape[1]
    t_grid = torch.linspace(0, 1, T)

    print(f'Initializing datasets:')

    # temp_ted = NamedTemporaryFile()
    # temp_teec = NamedTemporaryFile()

    # temp_trd = NamedTemporaryFile()
    # temp_trec = NamedTemporaryFile()
    files = []
    curtime = int(time.time()) % 100000

    try:
        directory = Path('/tmp/tensordict/')
        directory.mkdir(parents=False, exist_ok=True)
    except FileNotFoundError:
        directory = Path(os.path.join(parent_dir, 'saved_tensors'))
        directory.mkdir(parents=True, exist_ok=True)

    for i in range(4):
        files.append(os.path.join(directory, f'memmappedfile_{i}_{curtime}'))

    use_mem_mapped = True

    print('Ref count before NDDatasets: ', sys.getrefcount(solutions_train))

    grids = None
    train_dataset = NDDataset(solutions_train, extra_channels = [], # forcings_train
                              grids = None, dataset_index=0, use_mem_mapped=use_mem_mapped, 
                              file_to_memmap = '/workspace/backup/memmap_train')
    val_dataset   = NDDataset(solutions_test, extra_channels = [], # forcings_test
                              grids = None, dataset_index=0, use_mem_mapped=use_mem_mapped,
                              file_to_memmap = '/workspace/backup/memmap_val')

    #print('Ref count: ', sys.getrefcount(solutions_train))
    print(f'cur_solutions.shape {cur_solutions.shape}')
    sshape = cur_solutions.shape[1]
    fshape = cur_forcings.shape[1]
    del cur_solutions, cur_forcings
    del solutions_train, solutions_test, forcings_train, forcings_test
    gc.collect()
    print(f'Initialized datasets &, presumably, cleaned memory')

    for idx, sample in enumerate(train_dataset):
        sample_x = sample['x'].to('cuda')
        sample_y = sample['y'].to('cuda')

        if (idx % 100) == 0:
            print(f'Processing train sample {idx}: shapes are {sample_x.shape, sample_y.shape}, device: {sample_x.device}')
        inp_normalizer.partial_fit(sample_x)
        out_normalizer.partial_fit(sample_y)
    print('inp_normalizer.mean()', inp_normalizer.mean, 'inp_normalizer.std()', inp_normalizer.std) 
    print('out_normalizer.mean()', out_normalizer.mean, 'out_normalizer.std()', out_normalizer.std) 


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size) #, num_workers = min(4, os.cpu_count()))
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = batch_size) # , num_workers = min(4, os.cpu_count()))
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
                in_channels  = sshape + fshape + 2 * int(grids is not None)    # train_dataset.__getitem__(0).in_channels
                try:
                    out_channels = model_selection['params'][idx]['width']
                except KeyError:
                    out_channels = model_selection['params'][idx]['hidden_channels']

            elif idx == len(model_selection['model'])-1:
                in_channels  = model_selection['params'][idx]['hidden_channels']
                out_channels = sshape # dataset.out_channels
            else:
                try:
                    in_channels  = model_selection['params'][idx]['hidden_variable_codimension']
                    out_channels = model_selection['params'][idx]['hidden_variable_codimension']
                except:
                    in_channels  = model_selection['params'][idx]['hidden_channels']
                    out_channels = model_selection['params'][idx]['hidden_channels']

            model.append(submodel(in_channels  = in_channels,
                                  out_channels = out_channels,
                                  **model_selection['params'][idx]))
        model = tuple([[model[0],], model[1], [model[2],]])
    else:
        validateOperator(model_selection['model'], ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))

        print(f'dataset channels: in - {sshape + fshape + 2}, out - {sshape}')
        model = model_selection['model'](in_channels = sshape + fshape + 2 * int(grids is not None),
                                         out_channels = sshape,
                                         **model_selection['params'])

    now = datetime.now()

    trainer = Trainer()
    logger_filename = os.path.join(parent_dir, 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.setLogger(filename = logger_filename)

    model_name = None
    if model_name is not None:
        trainer.loadModel(model_name)
        print('Loaded model as ...')
    else:
        trainer.buildModel(model)

    if SCHEDULER_PARAMS['scheduler'] == 'cosine':
        SCHEDULER_PARAMS['max_cosine_lr_epochs'] = args.epochs_max

    loss1 = BalancedRelL2Loss()
    loss2 = FourierHFLoss()
    loss = [loss1, loss2]
    trainer.buildOptimizer(n_dim = 3,
                           params_scheduler = SCHEDULER_PARAMS,
                           params_opt = OPTIMIZER_PARAMS,
                           trainer_loss = loss)

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
