import torch
from torch.utils.data import DataLoader

from muno.data import UnitGaussianNormalizer, MultiphysicsUnitGaussianNormalizer
from muno.data.data.transforms.data_processors import DefaultDataProcessor


def get_channelwise_reduce_dims(batch_tensor):
    if not isinstance(batch_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(batch_tensor)}")
    if batch_tensor.ndim < 3:
        raise ValueError(f"Expected batched tensor [B, C, ...], got shape {tuple(batch_tensor.shape)}")

    return [0] + list(range(2, batch_tensor.ndim))


def fit_unit_gaussian_normalizer(loader, key, device="cuda", max_batches=None):
    first_batch = next(iter(loader))
    dim = get_channelwise_reduce_dims(first_batch[key])
    normalizer = UnitGaussianNormalizer(dim=dim)

    for batch_idx, batch in enumerate(loader):
        values = batch[key].to(device)

        fit_batch_size = max(values.shape[0], 2)
        normalizer.partial_fit(values, batch_size=fit_batch_size)

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return normalizer

def fit_multiphys_unit_gaussian_normalizer(loader, key, device='cuda', max_batches=None):
    first_batch = next(iter(loader))
    init_key = list(first_batch.keys())[0]

    dims = {i: get_channelwise_reduce_dims(subbatch[key]) for i, subbatch in first_batch.items()}

    normalizer = MultiphysicsUnitGaussianNormalizer(num=len(first_batch), dim=dims, key=key)
    # normalizer.to(device)
    for batch_idx, batch in enumerate(loader):
        # bvalues = {i: subbatch[key].to(device) for i, subbatch in batch.items()}

        # assert all([subbatch.shape[0] == bvalues[init_key].shape[0] for subbatch in bvalues.values()]), \
        #     'Batch has different sizes for different physics. Unsupported behavior.'
        fit_batch_size = max(batch[init_key][key].shape[0], 2)

        normalizer.partial_fit(batch, batch_size=fit_batch_size)

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    normalizer.to(device)
    return normalizer        


def build_data_processor(train_loader,
                         device="cuda",
                         normalize_x=True,
                         normalize_y=True,
                         max_fit_batches=None):
    in_normalizer = None
    out_normalizer = None

    if normalize_x:
        in_normalizer = fit_multiphys_unit_gaussian_normalizer(
            train_loader, key="x", device=device, max_batches=max_fit_batches, # device = device
        )
    if normalize_y:
        out_normalizer = fit_multiphys_unit_gaussian_normalizer(
            train_loader, key="y", device=device, max_batches=max_fit_batches, #  device = device
        )

    return DefaultDataProcessor(in_normalizer=in_normalizer,
                                out_normalizer=out_normalizer,
                                device=device)


def build_data_processors(train_loaders, config=None, device="cuda"):
    assert isinstance(train_loaders, DataLoader), \
        'Currently, build_data_processors requires train_loaders to be a single DataLoader'
    config = {} if config is None else config

    if not config.get("enabled", True):
        return None

    normalize_x = config.get("normalize_x", True)
    normalize_y = config.get("normalize_y", True)
    max_fit_batches = config.get("max_fit_batches")

    processor = build_data_processor(train_loaders,
                                     device=device,
                                     normalize_x=normalize_x,
                                     normalize_y=normalize_y,
                                     max_fit_batches=max_fit_batches)

    return processor
