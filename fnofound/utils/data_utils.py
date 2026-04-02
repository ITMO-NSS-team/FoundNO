import warnings
import os
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple, List, Callable

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
# from torch.utils.data.distributed import DistributedSampler

from .domains import Domain

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.title(title)
    plt.show()
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')

class SimpleDataset(Dataset):
    '''
    To be replaced, according to preprocessing tools
    '''
    def __init__(self, data: List[torch.Tensor], domain: Domain, inputs: List[List[torch.Tensor]] = None, # coords: List[torch.Tensor],
                 transform_x: Callable = None, transform_y: Callable = None, ic_ord: int = 1, 
                 incl_t: bool = False, device: str = 'cuda', dataset_index: int = 0): # , is_complex: bool = False
        self._device = device
        self._incl_t = incl_t
        self._dataset_index = dataset_index

        self._data = [tensor.to(self._device) for tensor in data] # Dimensionality has to be of [N, T, X_1, ...]
        self.check_inputs(inputs)

        # ic_num = ic_ord * data[0].shape[0].item() # * 2 if is_complex else ic_ord * data[0].shape[0].item()

        self.out_channels = data[0].shape[0]
        self.in_channels = len(inputs[0]) + domain.ndim + ic_ord * self.out_channels
        if not incl_t:
            self.in_channels -= 1

        print(f'Initializing dataset with {self.in_channels} - input, and {self.out_channels} output channels.')

        if inputs is not None:
            self._inputs = []
            for sample_idx, sample_inp_func in enumerate(inputs):
                if sample_idx == 0:
                    print(f'Having {len(sample_inp_func)}-inputs of shape {sample_inp_func[0].shape}')

                temp_inp = torch.stack([inp_tensor for inp_tensor in sample_inp_func]).to(self._device) # .unsqueeze(0)
                if sample_idx == 0:
                    print(f'Obtained {temp_inp.shape} stacked tensor') 

                self._inputs.append(temp_inp)
        else:
            self._inputs = []
        
        self._ic_ord = ic_ord
        self._domain = domain

        self.x_transformer = transform_x
        self.y_transformer = transform_y

    @staticmethod
    def check_inputs(inputs: List[List[torch.Tensor]]):
        if False:
            raise ValueError('Incorrect shapes of input tensors.')

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int]:
        outputs = self._data[index]

        repeat_shape = [1, outputs.shape[1],] + [1,] * (outputs.ndim - 2)

        input_tensors = [self._domain.get_grid(self._incl_t), self._inputs[index]] # .unsqueeze(0)

        # print(outputs.shape)
        if self._ic_ord > 0:
            input_tensors.append(outputs[:, 0:1, ...].repeat(repeat_shape))

        # print('Input tensors type: ', self._domain.get_grid().type(), self._inputs[index].type(), outputs[:, :, 0:1, ...].type())

        if self._ic_ord > 1:
            input_tensors.append((outputs[:, 1:2, ...] - outputs[:, 0:1, ...]) / self._domain.get_step(device = self._device)) # Move from getitem to init or separate preprocess method.
        if self._ic_ord > 2:
            warnings.warn('Equation demands more than 3 initial conditions. Their constructor has not yet benn constructed!')
            pass

        # Add inputs with forcing or specified boundary conditions

        # print(f'Shapes: ', [tensor.shape for tensor in input_tensors])
        inputs = torch.concat(input_tensors, dim = 0) # .permute(*self._permute_ord)
        if self.x_transformer is not None:
            inputs = self.x_transformer(inputs)

        # outputs = outputs.permute(*self._permute_ord)
        if self.y_transformer is not None:
            outputs = self.y_transformer(outputs)

        return {'x': inputs, 'y': outputs, 'eq_idx': self._dataset_index}
    

class Dataset2DAndTWithMasks(Dataset):
    """
    Ultra-optimized dataset с:
    1. Быстрой векторизованной генерацией масок
    2. Сохранением/загрузкой масок на диск
    3. Исправленными размерностями
    """
    def __init__(self, solutions, forcings, x_coords, y_coords, t_coords,
                 well_radius=3, random_sample_ratio=0.2, zero_threshold=1e-6,
                 cache_dir='./mask_cache', force_recompute=False):
        self.solutions = solutions
        self.forcings = forcings
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.t_coords = t_coords
        print(solutions.shape)
        self.T, self.C_u, self.H, self.W = solutions.shape[1:]

        self.well_radius = well_radius
        self.random_sample_ratio = random_sample_ratio
        self.zero_threshold = zero_threshold

        # Создаем директорию для кэша
        os.makedirs(cache_dir, exist_ok=True)

        # Имя файла кэша на основе параметров
        cache_filename = f'masks_N{len(solutions)}_T{self.T}_H{self.H}_W{self.W}_r{well_radius}_s{int(random_sample_ratio*100)}.pt'
        self.cache_path = os.path.join(cache_dir, cache_filename)

        # Загружаем или вычисляем маски
        if os.path.exists(self.cache_path) and not force_recompute:
            print(f"\n💾 Loading precomputed masks from cache: {cache_filename}")
            self.masks = torch.load(self.cache_path)
            print(f"✓ Loaded! Shape: {self.masks.shape}, Memory: ~{self._estimate_mask_memory():.2f} GB\n")
        else:
            print(f"\n🔧 Computing sampling masks for {len(self)} samples...")
            print(f"   Well radius: {well_radius}")
            print(f"   Random sample ratio: {random_sample_ratio:.1%}")

            start_time = time.time()
            self.masks = self._precompute_all_masks_fast()
            elapsed = time.time() - start_time

            print(f"✓ Masks computed in {elapsed:.1f}s! Memory: ~{self._estimate_mask_memory():.2f} GB")

            # Сохраняем в кэш
            print(f"💾 Saving masks to cache: {cache_filename}")
            torch.save(self.masks, self.cache_path)
            print(f"✓ Cached!\n")

    def _estimate_mask_memory(self):
        """Оценка памяти для масок."""
        if self.masks.dtype == torch.bool:
            bytes_per_element = 1
        else:
            bytes_per_element = 4  # float32

        total_bytes = self.masks.numel() * bytes_per_element
        return total_bytes / (1024**3)  # GB

    def _create_mask_for_sample_fast(self, forcings_sample):
        """
        БЫСТРАЯ векторизованная генерация маски для одного сэмпла.

        Args:
            forcings_sample: [T, 1, H, W]

        Returns:
            mask: [T, H, W] bool tensor
        """
        T, _, H, W = forcings_sample.shape

        # Находим скважины для всех временных шагов сразу
        well_mask = (torch.abs(forcings_sample[:, 0]) > self.zero_threshold).float()  # [T, H, W]

        # Расширяем область вокруг скважин для всех T одновременно
        kernel_size = 2 * self.well_radius + 1
        well_mask_expanded = well_mask.unsqueeze(1)  # [T, 1, H, W]

        dilated_well_mask = F.max_pool2d(
            well_mask_expanded.view(-1, 1, H, W),  # [T, 1, H, W]
            kernel_size=kernel_size,
            stride=1,
            padding=self.well_radius
        ).view(T, 1, H, W)[:, 0]  # [T, H, W]

        # Начинаем с well mask
        mask = (dilated_well_mask > 0.5)  # [T, H, W] bool

        # Добавляем случайные точки для каждого временного шага
        if self.random_sample_ratio > 0:
            for t in range(T):
                # Находим точки вне скважин
                non_well_points = ~mask[t]  # [H, W] bool

                if non_well_points.sum() > 0:
                    # Случайная выборка
                    n_total = non_well_points.sum().item()
                    n_sample = max(1, int(n_total * self.random_sample_ratio))

                    # Генерируем случайные индексы
                    non_well_indices = torch.nonzero(non_well_points, as_tuple=False)
                    perm = torch.randperm(len(non_well_indices))[:n_sample]
                    sampled_indices = non_well_indices[perm]

                    # Добавляем в маску
                    mask[t, sampled_indices[:, 0], sampled_indices[:, 1]] = True

        return mask  # [T, H, W] bool

    def _precompute_all_masks_fast(self):
        """Быстрое предвычисление всех масок с прогресс-баром."""
        masks = []

        for idx in tqdm(range(len(self)), desc="Computing masks", ncols=100):
            forcings_sample = self.forcings[idx]  # [T, 1, H, W]
            mask = self._create_mask_for_sample_fast(forcings_sample)
            masks.append(mask)

        print('masks:', len(masks), ' from ', self.solutions.shape)
        # Стакаем в bool tensor для экономии памяти
        return torch.stack(masks, dim=0)  # [N, T, H, W] bool

    def __len__(self):
        return self.solutions.shape[0]

    def __getitem__(self, idx):
        sol = self.solutions[idx]
        frc = self.forcings[idx]

        t_init = 0 #; np.random.randint(low=0, high=sol.shape[0])

        u0 = sol[t_init]
        u0_bcast = u0.unsqueeze(0).expand(self.T - t_init, -1, -1, -1)

        X = self.x_coords[None, None, :, :].expand(self.T - t_init, 1, -1, -1)
        Y = self.y_coords[None, None, :, :].expand(self.T - t_init, 1, -1, -1)

        inp = torch.cat([u0_bcast, frc[t_init:], X, Y], dim=1)
        out = sol[t_init:]
        frc_out = frc[t_init:]

        # Получаем предвычисленную маску
        mask = self.masks[idx, t_init:]  # [T-t_init, H, W] bool

        return {
            'x': inp.permute(1, 0, 2, 3),      # [5, T-t_init, H, W]
            'y': out.permute(1, 0, 2, 3),      # [2, T-t_init, H, W]
            'sources': frc_out.permute(1, 0, 2, 3),  # [1, T-t_init, H, W]
            'mask': mask,  # [T-t_init, H, W] bool - БЕЗ permute!
            'eq_idx' : 0
        }

class NDDataset(Dataset):
    def __init__(self, pred_fields: torch.Tensor, extra_channels: List[torch.Tensor], grids: List[torch.Tensor] = None,
                 dataset_index: int = 0, device: str = 'cuda'):
        self._dataset_index = dataset_index
        
        self._device = device

        self._spatial_dims = pred_fields.ndim - 2
        match pred_fields.ndim:
            case 3:
                self.B, self.C, self.T = pred_fields.shape
                self.X, self.Y = 1, 1
            case 4:
                self.B, self.C, self.T, self.X = pred_fields.shape
                self.Y = 1
            case 5: 
                self.B, self.C, self.T, self.X, self.Y = pred_fields.shape
            case 6: 
                raise NotImplementedError('3 spatial + time setup has not been implemented yet.')
            case _:
                raise ValueError("Dimensionality of data does not match problem: either less than 3, or higher than 6.")

        self._pred_fields = pred_fields  # [N, C, T, H, W]
        self._extra_channels = extra_channels
        if grids is None or (isinstance(grids, list) and len(grids) == 0):
            self._grids_used = False
            self._grids = []
        else:
            self._grids_used = True

            self._grids = grids
            assert len(self._grids) == self._spatial_dims - 1, 'Number of grids does not match problem requirements.'

    @property
    def out_channels(self):
        return self._pred_fields.shape[1]
    
    @property
    def in_channels(self):
        return self._pred_fields.shape[1] + len(self._extra_channels) + len(self._grids)

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        field = self._pred_fields[idx]   # [C, T, ...]
        
        extras = [channel[idx] for channel in self._extra_channels]    # all channels are like [C, T, ...]
        ic_bcast = field[:, 0:1]
        # print(f'ic_bcast is {ic_bcast.shape}')
        # plt.plot(ic_bcast.squeeze().cpu().detach().numpy())
        # plt.show()
        ic_bcast = ic_bcast.expand(-1, self.T, *([-1,] * (field.ndim-2)))
        

        if self._grids_used:
            match self._spatial_dims:
                case 1:
                    grids = []
                case 2:
                    grids = [self._grids[0][None, None, ...].expand(1, self.T, -1,),]
                case 3:
                    grids = [self._grids[0][None, None, ...].expand(1, self.T, -1, -1),
                             self._grids[1][None, None, ...].expand(1, self.T, -1, -1),]
        else:
            grids = []

        # print(f'Accessing ')
        inp = torch.cat([ic_bcast,] + extras + grids, dim = 0)  # [C, T, ...] ,
        out = field

        # Format for FNO: [C, T ...]
        return {'x': inp.to(self._device), 'y': out.to(self._device), 'eq_idx': self._dataset_index}