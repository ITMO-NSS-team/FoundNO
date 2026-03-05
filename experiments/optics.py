import os
import argparse
from datetime import datetime

from typing import List

import glob
import sys
import h5py

sys.path.append('.')

import torch

from neuralop.models import FNO

from foundno.utils.training_utils import load_files_hdf5, validate_operator, standartize

from foundno.utils.domains import Domain
from foundno.utils.data_utils import SimpleDataset
from foundno.utils.custom_trainer import Trainer, Logger

# from models.pecoda import PeCODANO
from foundno.models.mamba_fno import PostLiftMambaFNO
from foundno.models.localattn_exp import LocalAttnFNO
PeCODANO = None
OPTIMIZER_PARAMS = {'optimizer': "adam", 'lr': 1e-3}

SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 1e3}

ARGS = {'fno': {'model' : FNO,
                'params' : {'hidden_channels': 48,
                            'n_layers': 6,
                            'n_modes': [96, 96]}},
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

EXPNAME = 'optics_var_focus'

def load_h5py_file(file: h5py.File):
    sets = []
    for key in file.keys():
        raw_inputs   = torch.view_as_real(torch.from_numpy(file[key]['data']['inputs'][()]))
        lens_radius  = standartize(torch.full_like(input = raw_inputs[..., 0],
                                                   fill_value = file[key]['info']['lens_radius'][()]))
        fs_distance  = standartize(torch.full_like(input = raw_inputs[..., 0],
                                                   fill_value = file[key]['info']['fs_distance'][()]))
        lfl_distance = standartize(torch.full_like(input = raw_inputs[..., 0],
                                                   fill_value = file[key]['info']['lens_focal_length'][()]))
        outputs      = torch.view_as_real(torch.from_numpy(file[key]['data']['outputs'][()]))
        outputs[..., 0] = standartize(outputs[..., 0])
        outputs[..., 1] = standartize(outputs[..., 1])

        sets.append(([standartize(raw_inputs[..., 0]), standartize(raw_inputs[..., 1]), 
                      lens_radius, fs_distance, lfl_distance],
                     outputs.permute(2, 0, 1)))
    return sets

if __name__ == "__main__":
    print(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'fno') # , type = ascii
    parser.add_argument("--epochs_max", default = 1e4, type = int)

    parser.add_argument("--single_model_location", default = '') # , type = ascii
    parser.add_argument("--lift_model_location",   default = '') # , type = ascii
    parser.add_argument("--main_model_location",   default = '') # , type = ascii
    parser.add_argument("--proj_model_location",   default = '') # , type = ascii

    args = parser.parse_args()

    data_dir = os.path.join(parent_dir, 'data', EXPNAME) #'../data/heat'
    filepaths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))

    print(f'Working with {EXPNAME} problem: parent_dir is {parent_dir}, data_dir is {data_dir}.')

    # samples_pretrain = load_files_hdf5(filepaths, finetune = False, finetune_label = 'has_conv',
    #                                    to_standardize = True, original_shape = (51, 51))

    samples_pretrain = []
    for file in filepaths:
        with h5py.File(file, 'r') as f:
            samples_pretrain.extend(load_h5py_file(f))
    print(f'Loaded {len(samples_pretrain)} number of samples, with inputs: {len(samples_pretrain[0][0])},\
             of {samples_pretrain[0][1][0].shape}, outputs: {samples_pretrain[0][1].shape}')

    params = {'t': {'L': 1., 'n': samples_pretrain[0][1].shape[1]}, 'x': {'L': 1., 'n': samples_pretrain[0][1].shape[2]}}
    domain = Domain(params)

    batch_size = 20

    dataset = SimpleDataset(data = [sample[1] for sample in samples_pretrain], domain = domain,
                            inputs = [sample[0] for sample in samples_pretrain], ic_ord = 0)
    print('dataset._inputs.shape', dataset._inputs[0].shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

    model_selection = ARGS[args.model]
    validate_operator(model_selection['model'], ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))

    model = model_selection['model'](in_channels = dataset.in_channels,
                                     out_channels = dataset.out_channels,
                                     **model_selection['params'])

    now = datetime.now()

    trainer = Trainer()
    logger_filename = os.path.join(parent_dir, 'experiments', 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.set_logger(filename = logger_filename)

    trainer.build_model(model)
    trainer.build_optimizer(n_dim = 2,
                            params_scheduler = SCHEDULER_PARAMS,
                            params_opt = OPTIMIZER_PARAMS,
                            data_processor = None)

    trainer.to('cuda')
    trainer.train(loader, int(args.epochs_max))
    
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

    trainer.save_model(model_savefile)
