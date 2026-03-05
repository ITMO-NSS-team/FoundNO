import warnings

import torch
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