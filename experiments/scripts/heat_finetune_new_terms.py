import os
import argparse
from datetime import datetime

from typing import List

import glob
import sys

sys.path.append('.')

import torch

from neuralop.models import FNO

from fnofound.utils.training_utils import load_files_hdf5

from fnofound.utils.domains import Domain
from fnofound.utils.data_utils import SimpleDataset
# from utils.custom_trainer import Trainer, Logger
from fnofound.utils.custom_finetuner import FineTuner

OPTIMIZER_PARAMS = {'optimizer': "adam", 'lr': 1e-3}

SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 1e3}

EXPNAME = 'heat'

if __name__ == "__main__":
    print(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--single_model_location", default = '', type = ascii)
    parser.add_argument("--lift_model_location",   default = '', type = ascii)
    parser.add_argument("--main_model_location",   default = '', type = ascii)
    parser.add_argument("--proj_model_location",   default = '', type = ascii)

    parser.add_argument("--epochs_max", default = 1e4, type = int)

    args = parser.parse_args()

    if args.single_model_location == '':
        assert (args.lift_model_location != '' and
                args.main_model_location != '' and
                args.proj_model_location != ''), 'Early detection of incorrect locations of models'
    elif args.lift_model_location == '':
        assert args.single_model_location != '', 'Early detection of incorrect locations of models'
    else:
        raise AssertionError('Early detection of incorrect locations of models')

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

    now = datetime.now()

    trainer = FineTuner()
    logger_filename = os.path.join(parent_dir, 'experiments', 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.setLogger(filename = logger_filename)
    try:
        trainer.load_model(args.model_location)
    except:
        raise ValueError('Failed to load file, please, check location')
    trainer.buildOptimizer()

    trainer.to('cuda')
    trainer.train(loader, int(args.epochs_max))
    
    model = trainer.model

    model_savefile_base = os.path.join(parent_dir, 'experiments', 'finetuned_models')
    if trainer._single_model:
        filename = f'{EXPNAME}_{args.model}_{now.day}_{now.hour}_{now.minute}.pt'
        model_savefile = os.path.join(model_savefile_base, filename)
    else:
        filename_lift = f'{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.pt'
        filename_main = f'{EXPNAME}_{args.model}_main_{now.day}_{now.hour}_{now.minute}.pt'
        filename_proj = f'{EXPNAME}_{args.model}_proj_{now.day}_{now.hour}_{now.minute}.pt'

        model_savefile_lift = os.path.join(model_savefile_base, filename_lift)
        model_savefile_main = os.path.join(model_savefile_base, filename_main)
        model_savefile_proj = os.path.join(model_savefile_base, filename_proj)
        model_savefile = (model_savefile_lift, model_savefile_main, model_savefile_proj)

    trainer.saveModel(model_savefile)