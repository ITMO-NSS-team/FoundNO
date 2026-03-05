import os
import argparse
from datetime import datetime

from typing import List

import glob
import sys

sys.path.append('.')

import torch

from neuralop.models import FNO

from foundno.utils.training_utils import load_files_hdf5, validate_operator

from foundno.utils.domains import Domain
from foundno.utils.data_utils import SimpleDataset
from foundno.utils.custom_trainer import Trainer, Logger

from foundno.models.pecoda import PeCODANO
from foundno.models.mamba_fno import PostLiftMambaFNO
from foundno.models.localattn_exp import LocalAttnFNO

OPTIMIZER_PARAMS = {'optimizer': "adam", 'lr': 1e-3}

SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 1e3}

ARGS = {'fno': {'model' : FNO,
                'params' : {'hidden_channels': 16,
                            'n_layers': 6,
                            'n_modes': [64, 64]}},
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
    parser.add_argument("--epochs_max", default = 1e4, type = int)

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

    batch_size = 20

    dataset = SimpleDataset(data = [sample[1] for sample in samples_pretrain], domain = domain,
                            inputs = [[sample[0][0, ...]] for sample in samples_pretrain])
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
