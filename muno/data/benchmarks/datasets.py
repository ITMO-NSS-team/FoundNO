from typing import List, Dict, Callable
import random

import torch
from torch.utils.data import Dataset

class LazyCanonicalDataset(Dataset):
    def __init__(self, source, adapter, start=0, end=None):
        self.source = source
        self.adapter = adapter
        self.start = start
        self.end = len(self.source) if end is None else end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        if item >= self.__len__():
            item = random.randint(0, self.__len__() - 1)

        idx = self.start + item
        raw_sample = self.source.get_sample(idx)
        canonical_sample = self.adapter.canonize(raw_sample)
        return canonical_sample


class SlidingWindowCanonicalDataset(Dataset):
    def __init__(self, source, adapter, window_start_indices, start=0, end=None):
        self.source = source
        self.adapter = adapter
        self.window_start_indices = list(window_start_indices)
        self.start = start
        self.end = len(self.source) if end is None else end

        if not self.window_start_indices:
            raise ValueError("window_start_indices must contain at least one value")

    def __len__(self):
        return (self.end - self.start) * len(self.window_start_indices)

    def __getitem__(self, item):
        if item >= self.__len__():
            item = random.randint(0, self.__len__() - 1)

        trajectory_offset = item // len(self.window_start_indices)
        window_offset = item % len(self.window_start_indices)

        source_idx = self.start + trajectory_offset
        window_start = self.window_start_indices[window_offset]

        raw_sample = self.source.get_sample(source_idx)
        canonical_sample = self.adapter.canonize(
            raw_sample,
            window_start=window_start,
        )
        return canonical_sample


class IndexedCanonicalDataset(Dataset):
    def __init__(self, source, adapter, indices):
        self.source = source
        self.adapter = adapter
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if item >= self.__len__():
            item_old = item
            item = random.randint(0, self.__len__() - 1)
        else:
            item_old = item

        try:
            source_idx = self.indices[item]
        except IndexError:
            print(f'Error in getting a item from IndexedCanonicalDataset with len {self.__len__()} at idx {item} (from {item_old}).')
            raise(IndexError("list index out of range"))
        raw_sample = self.source.get_sample(source_idx)
        canonical_sample = self.adapter.canonize(raw_sample)
        return canonical_sample


class IndexedSlidingWindowCanonicalDataset(Dataset):
    def __init__(self, source, adapter, indices, window_start_indices):
        self.source = source
        self.adapter = adapter
        self.indices = list(indices)
        self.window_start_indices = list(window_start_indices)

        if not self.window_start_indices:
            raise ValueError("window_start_indices must contain at least one value")

    def __len__(self):
        return len(self.indices) * len(self.window_start_indices)

    def __getitem__(self, item):
        if item >= self.__len__():
            item = random.randint(0, self.__len__() - 1)

        trajectory_offset = item // len(self.window_start_indices)
        window_offset = item % len(self.window_start_indices)

        source_idx = self.indices[trajectory_offset]
        window_start = self.window_start_indices[window_offset]

        raw_sample = self.source.get_sample(source_idx)
        canonical_sample = self.adapter.canonize(
            raw_sample,
            window_start=window_start,
        )
        return canonical_sample

# class Resampler():

class MultiPhysicsDataset(Dataset):
    '''
    Multiphysics dataset, introduced to sample from multiple datasets simultaneously and in a balanced way. 
    '''
    def __init__(self, subdatasets: List[Dataset]) -> None:
        assert isinstance(subdatasets, list), \
            f'subdatasets must be sent as a LIST of datasets, instead got {type(subdatasets)}'
        assert all([isinstance(ds, Dataset) for ds in subdatasets]), \
            f'subdatasets must be sent as a list of DATASETS, instead got {[type(ds) for ds in subdatasets]}'
        
        self._datasets = subdatasets

    # def balance(self, balancing_method: Callable):
    #     lens = self._data_lens()
    #     for idx in range(len):
    #         self._datasets[idx] = balancing_method(self._datasets[idx], lens)

    @property
    def _data_lens(self) -> List[int]:
        return [len(ds) for ds in self._datasets]

    def __len__(self) -> int:
        return max(self._data_lens)

    def __getitem__(self, index) -> Dict[int, Dict[str, torch.Tensor]]:
        '''
        Returns syncronized samples from multiple datasets with specific physics.
        The results are presented as a dict with keys - problem_ids, values - dicts of X, Y and processors. 
        '''
        # idxs = [index % dl for dl in self._data_lens]
        sample = dict()
        for ds_idx, ds in enumerate(self._datasets):
            sample[ds_idx] = ds[index]  # self._datasets[ds_idx][elem]

        return sample