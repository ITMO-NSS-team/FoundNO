import torch
import argparse
from datetime import datetime

from typing import List
from dataclasses import dataclass

import h5py
import glob
import sys
import os

sys.path.append('.')

from neuralop.models import CODANO

from fnofound.utils.training_utils import load_files_hdf5, validate_operator

from fnofound.utils.domains import Domain
from fnofound.utils.data_utils import SimpleDataset
from fnofound.utils.custom_trainer import Trainer, Logger

from fnofound.models.pecoda import PeCODANO
from fnofound.models.mamba_fno import PostLiftMambaFNO3D, PostLiftMambaLifting # PostLiftMambaFNO, 
from fnofound.models.localattn_exp import LocalAttnFNO
from fnofound.models.fno import FNO
# from fnofound.models.fno import LocalAttnFNO

from neuralop.layers.channel_mlp import ChannelMLP

from fnofound.data.data.datasets.multiphysics_wrapper import load_data
from neuralop.training import setup
from neuralop.utils import get_project_root

from zencfg import make_config_from_cli
import sys

from fnofound.data.config.multiphysics_config import Default

DATA_PATH = '/media/anonymized/Data/huggingface/MultiFlow_poseidon/train/' # TODO: insert into separate arg from terminal

def get_MP_data(dim: str = 'txx'):
    config = make_config_from_cli(Default)
    config = config.to_dict()

    device, is_logger = setup(config)

    config.verbose = config.verbose and is_logger

    if config.verbose and is_logger:
        print(f"##### CONFIG #####\n")
        print(config)
        sys.stdout.flush()

    multiphysics_data = {}

    for physics_name in list(config.data.datasets.keys()):
        physics_config = config.data.datasets[physics_name]
        data_root = DATA_PATH

        print(f'Loading data from {data_root}')

        task_config_full = {
            'channel_dim': physics_config.channel_dim,
            'channels_squeezed': physics_config.channels_squeezed,
            'custom_in_channels': physics_config.custom_in_channels,
            # 'n_train': config.data.n_train,
            'n_train': physics_config.n_train,
            'n_val': physics_config.n_val,
            'n_tests': physics_config.n_tests,
            'train_samples': physics_config.train_samples,
            'val_samples': physics_config.val_samples,
            'tests_samples': physics_config.tests_samples,
            'batch_size': config.data.batch_size,
            'train_resolution': physics_config.train_resolution,
            'val_resolution': physics_config.val_resolution,
            'test_resolutions': physics_config.test_resolutions,
            'test_batch_sizes': physics_config.test_batch_sizes,
            'interpolate_mode': physics_config.interpolate_mode,
            'data_root': data_root,
            'encode_input': physics_config.encode_input,
            'encode_output': physics_config.encode_output,
            'physics_name': physics_config.file_name,
            # 'physics_train_name': physics_config.file_train,
            # 'physics_test_name': physics_config.file_test,
            'download_params': physics_config.download_params,
            'temporal_subsample': physics_config.temporal_subsample,
            'spatial_subsample': physics_config.spatial_subsample,
        }

        train_loader, val_loader, test_loaders, data_processor = load_data(**task_config_full)

        multiphysics_data[physics_name] = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loaders': test_loaders,
            'data_processor': data_processor
        }

    trained_datasets = [value['train_loader'] for value in multiphysics_data.values()]
    val_datasets = [value['train_loader'] for value in multiphysics_data.values()]
    test_datasets    = [value['test_loaders'] for value in multiphysics_data.values()]
    data_processers  = [value['data_processor'] for value in multiphysics_data.values()]

    return trained_datasets, val_datasets, test_datasets, data_processers


def balanced_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, zero_threshold: float = 1e-6, eps: float = 1e-6):
    total_loss = 0.0
    C = pred.shape[1]
    for c in range(C):
        p = pred[:, c:c + 1]
        t = target[:, c:c + 1]
        mask = torch.abs(t) > zero_threshold
        if mask.sum() == 0:
            continue
        diff_norm = torch.norm((p - t) * mask)
        target_norm = torch.norm(t * mask) + eps
        total_loss += diff_norm / target_norm

    return total_loss / C if C > 0 else torch.tensor(0.0, device=pred.device)

OPTIMIZER_PARAMS = {'optimizer': "adam", 'lr': 1e-3, "loss": balanced_rel_l2_loss}

# SCHEDULER_PARAMS = {'scheduler': 'cosine', 'max_cosine_lr_epochs': 1e3}
SCHEDULER_PARAMS = {'scheduler': 'reducelr', 'patience': 4, 'factor': 0.5, 'min_lr': 1e-6}


ARGS = {'fno': {'model': FNO,
                'params': {'hidden_channels': 16,
                           'n_layers': 4,
                           'n_modes': [16, 32, 32]}},
        'mambafno': {'model': PostLiftMambaFNO3D,
                     'params': {'modes': (16, 32, 32),
                                'width': 32,
                                'n_layers': 4,
                                'use_mamba_kwargs': None,
                                'mamba_fallback_kernel': 9}},
        'localattnfno': {'model': LocalAttnFNO,
                         'params': {'width': 64,
                                    'n_local_layers': 2,
                                    'n_heads': 4,
                                    'window_size': 127}},
        'pecoda': {'model': PeCODANO,
                   'params': {'hidden_variable_codimension': 16,
                              'n_layers': 1,
                              'n_modes': [[6, 42, 42], ]}},
        'pecoda_split': {'model': [PostLiftMambaLifting, PeCODANO, ChannelMLP],
                         'params': [{'width': 16,
                                     'use_mamba_kwargs': None,
                                     'mamba_fallback_kernel': 9,
                                     'padding': 0,
                                     'n_dim': 2},
                                    {'hidden_variable_codimension': 16,
                                     'n_layers': 4,
                                     'n_modes': [[16, 16], [16, 16], [16, 16], [16, 16]],  # [8, 32, 32]
                                     'lifting_channels': None,
                                     'projection_channels': None
                                    },
                                    {'hidden_channels': 16,
                                     'n_layers': 2,
                                     'n_dim': 2,
                                     'non_linearity': torch.nn.functional.gelu}]},
        'pecoda_mlp': {'model': [ChannelMLP, PeCODANO, ChannelMLP],
                       'params': [{'hidden_channels': 16,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu},
                                  {'hidden_variable_codimension': 16,
                                   'n_layers': 4,
                                   'n_modes': [[16, 16], [16, 16], [16, 16], [16, 16]],  # [8, 32, 32]
                                   'lifting_channels': None,
                                   'projection_channels': None
                                #    'enable_cls_token': True
                                   },
                                  {'hidden_channels': 16,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu}]},
        'adapted_fno': {'model': [PostLiftMambaLifting, FNO, ChannelMLP],
                       'params': [{'width': 16,
                                   'use_mamba_kwargs': None,
                                   'mamba_fallback_kernel': 9,
                                   'padding': 0,
                                   'n_dim': 3},
                                  {'hidden_channels': 16,
                                   'n_layers': 4,
                                   'n_modes': [6, 16, 16],  # [8, 32, 32]
                                   'disable_lifting_and_projection': True 
                                   },
                                  {'hidden_channels': 16,
                                   'n_layers': 2,
                                   'n_dim': 3,
                                   'non_linearity': torch.nn.functional.gelu}]},
        'adapted_coda': {'model': [PostLiftMambaLifting, CODANO, ChannelMLP],
                       'params': [{'width': 16,
                                   'use_mamba_kwargs': None,
                                   'mamba_fallback_kernel': 9,
                                   'padding': 0,
                                   'n_dim': 2},
                                  {'n_layers': 4,
                                   'n_modes': [[32, 32], [32, 32], [32, 32], [32, 32]],
                                   'n_heads': [1, 1, 1, 1],
                                   'per_layer_scaling_factors': [[1, 1], [1, 1], [1, 1], [1, 1]],
                                   'attention_scaling_factors': [1., 1., 1., 1.,],
                                   'lifting_channels': None, 
                                   'projection_channels': None,
                                   'use_horizontal_skip_connection': True,
                                   'horizontal_skips_map': {2: 0, 3: 1}}, #'hidden_channels': 16,
                                #    'n_layers': 4,
                                #    'n_modes': [16, 16],  # [8, 32, 32]
                                #    'disable_lifting_and_projection': True 
                                   
                                  {'hidden_channels': 16,
                                   'n_layers': 2,
                                   'n_dim': 2,
                                   'non_linearity': torch.nn.functional.gelu}]},                                   

        } # CODANO


EXPNAME = 'gpn_multiphys'

@dataclass
class ExpSetup:
    model: str = 'adapted_fno' # 'adapted_coda' #  'adapted_coda' # 'pecoda_split' # 'adapted_fno' # 
    epochs_max: int = 2e3
    single_model_location: str = ''
    lift_model_location:   str = ''
    main_model_location:   str = ''
    proj_model_location:   str = ''

if __name__ == "__main__":
    print(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default = 'fno') # , type = ascii
    # parser.add_argument("--epochs_max", default = 1e2, type = int)

    # parser.add_argument("--single_model_location", default = '') # , type = ascii
    # parser.add_argument("--lift_model_location",   default = '') # , type = ascii
    # parser.add_argument("--main_model_location",   default = '') # , type = ascii
    # parser.add_argument("--proj_model_location",   default = '') # , type = ascii

    args = ExpSetup() # parser.parse_args()

    # data_dir = os.path.join(parent_dir, 'data', EXPNAME) #'../data/heat'
    # filepaths = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))

    # filepath = r"/home/anonymized/Documents/gpn/dataset_on_off_2/cleaned_data_on_off_2.hdf5"
    # with h5py.File(filepath, "r") as f:
    #     solutions = torch.tensor(f['dataset'][:200, :200], dtype=torch.float32)  # [T, N, 2, H, W]
    #     forcings  = torch.tensor(f['source'][:200, :200], dtype=torch.float32)   # [T, N, 1, H, W]

    # print(f'Loaded data with shape {solutions.shape}')
    # # Shuffle samples
    # N = solutions.shape[1]
    # perm = torch.randperm(N)
    # solutions = solutions[:, perm]
    # forcings = forcings[:, perm]

    # # To [N, T, C, H, W]
    # solutions = solutions.permute(1, 2, 0, 3, 4)
    # forcings = forcings.permute(1, 2, 0, 3, 4)
    # print(forcings.shape)
    # print(solutions.shape)
    # # Coordinates
    # H, W = 113, 134
    # x = torch.linspace(0, 1, W)
    # y = torch.linspace(0, 1, H)
    # X_grid, Y_grid = torch.meshgrid(y, x, indexing='ij')
    # T = 37
    # t = torch.linspace(0, 1, T)

    # params = {'t': {'L': t[-1], 'n': t.shape[0]},
    #           'y': {'L': y[-1], 'n': y.shape[0]},
    #           'x': {'L': x[-1], 'n': x.shape[0]}}
    # domain = Domain(params)

    # print(f'Solutions shape {solutions.shape}, forcings shape {forcings.shape}')

    # batch_size = 1 #20
    # dataset = SimpleDataset(data = [solutions[i] for i in range(solutions.shape[0])], domain = domain, # :i+1
    #                         inputs = [forcings[i, ...] for i in range(forcings.shape[0])])
    # del solutions, forcings

    # loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

    train_datasets, val_datasets, _, processors = get_MP_data()
    assert len(train_datasets) == len(val_datasets), 'Mismatching lengths of datasets.'
    # print('---------------------------- train_datasets: ----------------------------')
    # print(f'train_datasets type: {type(train_datasets)}, len: {len(train_datasets)};')

    for idx, _ in enumerate(processors):
        # print(f'train_datasets {idx}: {type(train_datasets[idx])}')
        # print(f'val_datasets {idx}: {type(val_datasets[idx])}')

        train_datasets[idx].dataset.set_idx(idx)
        val_datasets[idx].dataset.set_idx(idx)

        # for valloader in val_datasets[idx].values():
        #     valloader.dataset.set_idx(idx)

    model_selection = ARGS[args.model]

    # if isinstance(model_selection['model'], (tuple, list)):
    #     model = list()
    #     for idx, submodel in enumerate(model_selection['model']):
    #         if idx == 0:
    #             liftings = []
    #             for data_idx, set in enumerate(train_datasets):
    #                 in_channels = set.in_channels
    #                 out_channels = model_selection['params'][idx]['width']
    #                 liftings.append(submodel(in_channels=in_channels,
    #                                          out_channels=out_channels,
    #                                          **model_selection['params'][idx]))
    #             model.append(liftings)
    #
    #         elif idx == len(model_selection['model']) - 1:
    #             projections = []
    #             for data_idx, set in enumerate(train_datasets):
    #                 in_channels = model_selection['params'][idx]['hidden_channels']
    #                 out_channels = set.out_channels
    #                 projections.append(submodel(in_channels=in_channels,
    #                                             out_channels=out_channels,
    #                                             **model_selection['params'][idx]))
    #             model.append(projections)
    #
    #         else:
    #             in_channels = model_selection['params'][idx]['hidden_variable_codimension']
    #             out_channels = model_selection['params'][idx]['hidden_variable_codimension']
    #             model.append(submodel(in_channels=in_channels,
    #                                   out_channels=out_channels,
    #                                   **model_selection['params'][idx]))
    #
    #     model = tuple([model[0], model[1], model[2]])
    # else:
    #     assert len(train_datasets) == 1, 'Trying to train a single lift-proj model on multiple datasets'
    #     validate_operator(model_selection['model'],
    #                       ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))
    #
    #     print(f'dataset channels: in - {train_datasets[0].in_channels}, out - {train_datasets[0].out_channels}')
    #     model = model_selection['model'](in_channels=train_datasets[0].in_channels,
    #                                      out_channels=train_datasets[0].out_channels,
    #                                      **model_selection['params'])

    if isinstance(model_selection['model'], (tuple, list)):
        model = list()
        if (('hidden_variable_codimension' in model_selection['params'][1].keys()) or
            ('hidden_channels' in model_selection['params'][1].keys())):
            key = 'hidden_variable_codimension' if 'hidden_variable_codimension' in model_selection['params'][1].keys() else 'hidden_channels'
        else:
            key = None

        for idx, submodel in enumerate(model_selection['model']):
            # print(f'model_selection["params"][idx] {model_selection["params"][idx]}')

            if idx == 0:
                liftings = []
                for data_idx, subset in enumerate(train_datasets):

                    in_channels = subset.dataset.in_channels
                    print(f'IN CHANNELS: {in_channels}')
                    if key is None:
                        local_key = 'width' if 'width' in model_selection['params'][0].keys() else 'hidden_channels'
                        out_channels =  model_selection['params'][0][local_key]
                    else:
                        out_channels =  model_selection['params'][1][key]
                    print(f'Lifting out channels: {out_channels}')
                    liftings.append(submodel(in_channels=in_channels,
                                             out_channels=out_channels,
                                             **model_selection['params'][idx]))
                    # print(f'dataset {data_idx} channels: in - {in_channels}, out - {out_channels}')

                model.append(liftings)
            elif idx == len(model_selection['model'])-1:
                projections = []
                for data_idx, subset in enumerate(train_datasets):
                    if key is None:
                        local_key = 'width' if 'width' in model_selection['params'][-1].keys() else 'hidden_channels'
                        in_channels =  model_selection['params'][-1][local_key]
                    else:
                        in_channels =  model_selection['params'][1][key]

                    out_channels = subset.dataset.out_channels
                    print(f'Lifting out channels: {out_channels}')

                    projections.append(submodel(in_channels  = in_channels,
                                                out_channels = out_channels,
                                                **model_selection['params'][idx]))
                model.append(projections)

            else:
                # key = 'hidden_variable_codimension' if 'hidden_variable_codimension' in model_selection['params'][idx].keys() else 'hidden_channels'
                if key is None:
                    model.append(submodel(**model_selection['params'][idx]))
                else:
                    in_channels  = model_selection['params'][idx][key]
                    out_channels = model_selection['params'][idx][key]
                    model.append(submodel(in_channels  = in_channels,
                                          out_channels = out_channels,
                                          **model_selection['params'][idx]))

        model = tuple([model[0], model[1], model[2]])
    else:
        assert len(train_datasets) == 1, 'Trying to train a single lift-proj model on multiple datasets'
        validate_operator(model_selection['model'],
                          ['in_channels', 'out_channels'] + list(model_selection['params'].keys()))

        print(f'dataset channels: in - {train_datasets[0].in_channels}, out - {train_datasets[0].out_channels}')
        model = model_selection['model'](in_channels=train_datasets[0].dataset.in_channels,
                                         out_channels=train_datasets[0].dataset.out_channels,
                                         **model_selection['params'])

    now = datetime.now()

    trainer = Trainer()
    logger_filename = os.path.join(parent_dir, 'experiments', 'logs',
                                   f'log_{EXPNAME}_{args.model}_lift_{now.day}_{now.hour}_{now.minute}.log')
    trainer.setLogger(filename=logger_filename)

    trainer.buildModel(model)
    trainer.buildOptimizer(n_dim=3,
                           params_scheduler=SCHEDULER_PARAMS,
                           params_opt=OPTIMIZER_PARAMS,
                           # data_processor=None
                           )

    trainer.to('cuda')
    trainer.train(train_datasets, val_datasets, int(args.epochs_max), processors)

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
