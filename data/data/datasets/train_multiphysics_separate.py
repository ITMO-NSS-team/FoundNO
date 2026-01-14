from neuralop.data.datasets.multiphysics_wrapper import load_data
from neuralop.training import setup
from neuralop.utils import get_project_root

from zencfg import make_config_from_cli
import sys

sys.path.insert(0, '../')
from config.multiphysics_config import Default


def main():
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
        data_root = get_project_root() / config.data.folder

        task_config_full = {
            'n_train': config.data.n_train,
            'n_tests': physics_config.n_tests,
            'batch_size': config.data.batch_size,
            'train_resolution': physics_config.train_resolution,
            'test_resolutions': physics_config.test_resolutions,
            'test_batch_sizes': physics_config.test_batch_sizes,
            'interpolate_mode': physics_config.interpolate_mode,
            'data_root': data_root,
            'encode_input': physics_config.encode_input,
            'encode_output': physics_config.encode_output,
            'physics_name': physics_name,
            'download_params': physics_config.download_params,
            'temporal_subsample': physics_config.temporal_subsample,
            'spatial_subsample': physics_config.spatial_subsample,
        }

        train_loader, test_loaders, data_processor = load_data(**task_config_full)

        multiphysics_data[physics_name] = {
            'train_loader': train_loader,
            'test_loaders': test_loaders,
            'data_processor': data_processor
        }
    

if __name__ == '__main__':
    main()
