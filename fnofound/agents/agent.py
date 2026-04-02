import os
import sys

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Dict
from dataclasses import dataclass
from warnings import warn

import glob

import numpy as np
import torch
from neuralop.layers.fno_block import FNOBlocks
import torch.utils

from fnofound.data.data.transforms.base_transforms import Transform
from fnofound.data.data.transforms.data_processors import DefaultDataProcessor
from fnofound.data import UnitGaussianNormalizer

from fnofound.models.fno import FNO
from fnofound.models.local_no import LocalNO

from fnofound.models.coda import CODANO
from fnofound.models.pecoda import PeCODANO
from fnofound.models.mamba_fno import PostLiftMambaFNO3D, PostLiftMambaLifting
from fnofound.models.localattn_exp import LocalAttnFNO
from neuralop.layers.channel_mlp import ChannelMLP

@dataclass
class Message:
    '''
    Message of fixed type to send between agents: 
    Format as "I am agent ``ID'': to followup agents ``followup_IDs'': check your readiness"
    '''    
    ID: int
    followup_IDs: List[int] # or change to List[Tuple[int, bool]] as 

@dataclass
class CalcMessage(Message):
    '''
    Message of fixed type to send between agents: 
    approved format as "I am agent ``ID'': to followup agents ``followup_IDs'': 
                        check your readiness, if everything ready - commence.
                        Calculated channels ``C_0, C_1, ...'' on horizon ``[(t_0, t_1), (x1_0, x1_1), ...]''."
    '''
    channel  : List[int]
    horizon  : List[Tuple[int]] = None

@dataclass
class ChannelExtensionMessage(Message):
    '''
    Message of fixed type to send between agents: 
    approved format as "I am agent ``ID'': to followup agents ``followup_IDs'': 
                        check your readiness, if everything ready - commence.
                        I have added channel/channels N/N to M."
    '''
    channels: Union[int, Tuple[int]]


class EnvMeta(type):
    _env_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        if cls not in cls._env_instances:
            instance = super().__call__(*args, **kwargs)
            cls._env_instances[cls] = instance
            
        return cls._env_instances[cls]
    
    def reset(self):
        self._env_instances = {}


class NeuralOpSystemEnvironment(metaclass = EnvMeta):
    def __init__(self, channel_meta: List[int]):
        self._cores = {} # Dict of cores: represented as {key (str) : core (FNOBlock or similar models)}
        self._channel_meta = channel_meta  

    def setTensor(self, initial_tensor: torch.Tensor) -> None:
        '''
        Object for an environment for neuralop-based agents. Exists as a large torch.Tensor, 
        where initial conditions are gradually replaced with predictions.

        Args:
            initial_tensor: torch.Tensor, 
                Tensor of inputs (initial conditions & external channels, e.g. forcings, grIDs, etc.)
            channel_meta: List[int],
                Numbers of variables, set with the IC in channels. From 0 and up. -1 marks external channel.

        '''
        assert len(self._channel_meta) == initial_tensor.shape[1], \
               f'Mismatching channels and descriptions {self._channel_meta} vs {initial_tensor.shape}.'

        self._env = initial_tensor
        self._env_shape = initial_tensor.shape

    def setCores(self, cores: Dict[str, torch.nn.Module]):
        assert isinstance(cores, dict) and all(isinstance(key, str) for key in cores.keys()), 'Incorrect cores passed!'
        self._cores = cores

    def callCore(self, core_ID: str, inp: torch.Tensor):
        return self._cores[core_ID](inp)

    def access(self, channels: List[Union[Tuple[int], List[int], int]] = None) -> torch.Tensor: # Send slices as tuples of ints
        if channels is not None:
            chan_indexes = []
            for elem in channels:
                if isinstance(elem, int):
                    chan_indexes.append(elem)
                elif isinstance(elem, tuple):
                    chan_indexes += list(elem)
                elif isinstance(elem, list):
                    chan_indexes += elem
                else:
                    raise NotImplementedError('Got incorrect index, expected int, or list/tuple of ints.')

            return self._env[:, tuple(chan_indexes), ...]
        else:
            return self._env
        
    def modifyChannels(self, channels: List[Union[Tuple[int], List[int], int]], inp: torch.Tensor) -> None:
        # Add check of inp shape
        if isinstance(channels, int):
            # inp shape has to be [B, 1, T, ...]
            self._env[:, channels, ...] = inp
        else:
            for inner_chan_idx, channel in enumerate(channels):
                self._env[:, channel, ...] = inp[:, inner_chan_idx, ...]

    def concatNewChannels(self, new_channels: torch.Tensor) -> None:
        '''
        Function to add new channels, may be set not by batch (e.g. as [C x T x X x ...])
        '''
        if new_channels.shape[0] == 1 or new_channels.ndim < self._env.ndim:
            # Maybe add asserts

            self._channel_meta += [-1,] * new_channels.shape[0]

            repeat_shape = [i for i in self._env.shape]
            repeat_shape[0] =  self._env.shape[0] * repeat_shape[0]

            self._env = torch.concat([self._env, 
                                      torch.unsqueeze(new_channels, 0).expand(repeat_shape)], dim = 1)
        else:
            self._channel_meta += [-1,] * new_channels.shape[1]
            self._env = torch.concat([self._env, new_channels], dim = 1)

    def getVarState(self) -> torch.Tensor:
        indexes = tuple([i for i in range(self._env.shape[1]) if self._channel_meta[i] != -1])
        return self._env[:, indexes, ...]

    def saveEnv(self, files_dir: str, file_example_name: str = 'core_'):
        for core_key, core in self._cores.items():
            torch.save(obj = core, f = os.path.join(files_dir, file_example_name + core_key + '.pt'))

        # TODO: if neccessary, add additional saving of channels info, metadata and so on

    def loadEnv(self, files_dir: str, file_example_name: str = 'core_'): #  in_channels: int = 5, out_channels: int = 2
        if len(self._cores) != 0:
            warn("Some cores are already present in environment")
        self._cores = {}

        # data_dir = os.path.join(file_dir, 'data', EXPNAME) #'../data/heat'
        filepaths = sorted(glob.glob(os.path.join(files_dir, '*.pt')))        
        for filepath in filepaths:
            try: # Better check, if the state_dict is saved or the entire model
                core_params = torch.load(filepath, map_location="cuda")
                
                # TODO: change core declaration to have at least some level of flexibility
                try:
                    core = FNO(in_channels = 1, # dummy value, no in_channels really required
                            out_channels = 1, # dummy value, no in_channels really required
                            hidden_channels = 32,
                            n_layers = 4,
                            n_modes = [20, 42, 42], 
                            disable_lifting_and_projection = True)
                    core.load_state_dict(core_params)
                except:
                    core = LocalNO(in_channels = 1, # dummy value, no in_channels really required
                                out_channels = 1, # dummy value, no in_channels really required
                                hidden_channels =32,
                                default_in_shape = (113, 134),
                                n_layers = 4,
                                n_modes = (20, 42, 42),
                                radius_cutoff = 0.01,
                                lifting_channel_ratio = 0,
                                projection_channel_ratio = 0,
                                factorization = 'tucker', 
                                rank = 0.05,
                                implementation = 'factorized',
                                use_channel_mlp = True,
                                channel_mlp_dropout = 0.1)
                    core.load_state_dict(core_params)
            except: #  Better check, if the state_dict is saved or the entire model
                core = core_params
                print(f'type(core) is {type(core)}')

            for param in core.parameters():
                param.requires_grad = False

            ID = filepath.replace(files_dir, '').replace('.pt', '').replace(file_example_name, '').replace(os.path.sep, '')
            print(ID)

            self._cores[ID] = core


def checkShapeMatching(inp: torch.Tensor, other_tensors: List[torch.Tensor]) -> bool:
    if len(other_tensors) == 0:
        return True
    else:
        if inp.ndim != other_tensors[0].ndim:
            return False
        for idx in range(inp.ndim):
            if idx == 1:
                continue # Here we assert, that inputs may have different number of channels
            if inp.shape[idx] != other_tensors.shape[idx]:
                return False

        return True


class AbstractAgent(object): # ABC
    _CONTAINS_PARAMETERS = False

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def passMessage(self):
        raise NotImplementedError('Trying to call passMessage of an abstract agent.')

    @abstractmethod
    def getInfoDict(self):
        raise NotImplementedError('Trying to call getInfoDict of an abstract agent.')

    @abstractmethod
    def checkReadiness(self):
        raise NotImplementedError('Trying to call checkReadiness of an abstract agent.')

    @abstractmethod
    def resetReadiness(self):
        raise NotImplementedError('Trying to call resetReadiness of an abstract agent.')

    @abstractmethod
    def resetSuccessors(self):
        raise NotImplementedError('Trying to call resetSuccessors of an abstract agent.')

    @abstractmethod
    def addSuccessors(self, *args, **kwargs):
        raise NotImplementedError('Trying to call addSuccessors of an abstract agent.')

    @abstractmethod
    def saveAgent(self, *args, **kwargs):
        raise NotImplementedError('Trying to call saveAgent of an abstract agent.')


class BasicAgent(AbstractAgent): # torch.nn.Module
    _CONTAINS_PARAMETERS = False

    def __init__(self, ID: int,
                 env: NeuralOpSystemEnvironment,
                 predecessors : List[Tuple[int, AbstractAgent]],
                 successors   : List[Tuple[int, AbstractAgent]] = None):
        print(type(self))
        self._ID = ID

        self.env = env
        self._predecessors = {pred[0] : pred[1] for pred in predecessors}

        if successors is None:
            self._successors = {}
        else:
            self._successors = successors

        self._predecessors_exec = {ID: False for ID in self._predecessors.keys()}
        for predecessor in self._predecessors.values():
            print(f'Added successor {self._ID} to the agent {predecessor._ID}')
            predecessor.addSuccessors((self._ID, self))

    def getInfoDict(self) -> dict:
        predecessors = {} if self._predecessors is None else self._predecessors
        successors   = {} if self._successors   is None else self._successors
        print(f'predecessors: {predecessors}, successors {successors}')

        return {'meta'         : ['b'],
                'ID'           : self._ID,
                'predecessors' : [pred for pred in predecessors.keys()],
                'successors'   : [scs for scs in successors.keys()]}

    def checkReadiness(self) -> bool:
        return all(self._predecessors_exec.values())

    def resetReadiness(self) -> None:
        for key in self._predecessors_exec.keys():
            self._predecessors_exec[key] = False

    def setPredecessorAsExecuted(self, pred_ID: int):
        if pred_ID in self._predecessors_exec.keys():
            self._predecessors_exec[pred_ID] = True
        else:
            warn(f"Operator with ID {pred_ID} is missing from the predecessors of operator {self._ID}")

    def resetSuccessors(self) -> None:
        self._successors = {}

    def addSuccessors(self, successors: Union[Tuple[int, AbstractAgent], List[Tuple[int, AbstractAgent]]]) -> None:
        if isinstance(successors, tuple):
            successors = [successors,]

        for successor in successors:
            if successor[0] in self._successors.keys():
                raise RuntimeError('ID of successor already in the operator')
            self._successors[successor[0]] = successor[1]      

    def loadAgent(self, *args, **kwargs):
        pass

    def saveAgent(self, *args, **kwargs):
        pass


class ICLoaderMeta(type):
    _ic_loaders_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        if cls not in cls._ic_loaders_instances:
            instance = super().__call__(*args, **kwargs)
            cls._ic_loaders_instances[cls] = instance
            
        return cls._ic_loaders_instances[cls]
    
    def reset(self):
        self._ic_loaders_instances = {}


class InitialConditionsAgent(AbstractAgent, metaclass = ICLoaderMeta): #  BasicAgent
    _ID = 0
    _CONTAINS_PARAMETERS = False

    def __init__(self, 
                 env: NeuralOpSystemEnvironment,
                #  channel_info: List[int],
                #  additional_channels: torch.Tensor,
                 predecessors: Dict[int, AbstractAgent] = None):
        '''

        Args:
            inputs (Dict[int, torch.Tensor]): dict of tensors of initial conditions [B, C*, 1, X, Y, ...]
                keys - IDs of the required operators
            additional_channels (torch.Tensor) : torch Tensor of additional pre-defined channels (e.g. forcings)
                [B, C_add, T, X, Y, ...]

        C* - number of channels per variable in specific IC tensor. Typically, 1.

        '''
        self.env = env

        # self._channel_info = channel_info
        # self._additional_channels = additional_channels
        self._predecessors = predecessors

        self.resetSuccessors()

    def getInfoDict(self) -> dict:
        predecessors = {} if self._predecessors is None else self._predecessors
        successors   = {} if self._successors   is None else self._successors
            
        print(f'predecessors: {predecessors}, successors {successors}')
        print(f'from predecessors: {self._predecessors}, successors {self._successors}')


        return {'meta'         : ['i'],
                'ID'           : self._ID,
                'predecessors' : [pred for pred in predecessors.keys()],
                'successors'   : [scs for scs in successors.keys()]}        

    def resetSuccessors(self) -> None:
        self._successors = {}

    def addSuccessors(self, successors: Union[Tuple[int, AbstractAgent], List[Tuple[int, AbstractAgent]]]) -> None:
        if isinstance(successors, tuple):
            successors = [successors,] 

        for successor in successors:
            if successor[0] in self._successors.keys():
                raise RuntimeError('ID of successor already in the operator')
            self._successors[successor[0]] = successor[1]

    # def formInputTensor(self, ic) -> Dict[int, torch.Tensor]:
    #     # Maybe add better memory usage
    #     if isinstance(ic, torch.Tensor):
    #         tensors = [ic,]
    #     else:
    #         tensors = []
    #         for ID, tensor in ic:
    #             tensors.append(tensor)

    #     tensors.append(self._additional_channels)

    #     # print('shapes: ', [tensor.shape for tensor in tensors])
    #     return torch.cat(tensors, dim = 1)

    def passMessage(self, ic: torch.Tensor, inc_message: Message = None) -> bool: # , ic: Dict[int, torch.Tensor]
        # Let's assume, that the inc. message is None, i.e. this is the "first executed agent"
        exec_scs_agents = True

        self.env.setTensor(ic)  # self._initial_conds self._channel_info

        message = Message(self._ID, list(self._successors.keys()))
        for scs_ID, scs_agent in self._successors.items():
            exec_flag = scs_agent.passMessage(message)
            exec_scs_agents = exec_scs_agents and exec_flag

        assert exec_scs_agents, 'Some subagent did not get required data, thus failed to execute neural op.'
        return True
    
    def loadAgent(self, *args, **kwargs):
        pass

    def saveAgent(self, *args, **kwargs):
        pass


class NeuralOperatorAgent(BasicAgent, torch.nn.Module):
    _CONTAINS_PARAMETERS = True
    _SAVE_LOAD_PARAMS = {}

    def __init__(self,
                 ID: int,
                 arg_channels    : List[int], # ordered ids of channels
                 target_channels : List[int],
                 env             : NeuralOpSystemEnvironment,
                 adapters        : Tuple[torch.nn.Module, Tuple[torch.nn.Module]],
                 core            : str, # Tuple[int, Union[FNOBlocks, torch.nn.Module]],
                 predecessors    : List[Tuple[int, AbstractAgent]],
                 successors      : List[Tuple[int, AbstractAgent]] = None,
                 preprocessors   : Union[Tuple[Transform], DefaultDataProcessor] = None,
                 device          : str = 'cuda'):
        BasicAgent.__init__(self, ID=ID, env=env, predecessors=predecessors, successors=successors) # super()
        torch.nn.Module.__init__(self) # super()

        self.preprocessors = preprocessors

        # print('-'*10 + ' Adapters ' + '-'*10)
        # print(adapters)

        self._adapters = adapters
        self._core = core
        self._channel_ord = arg_channels
        self._target_channels = target_channels
        # Add additional checks on dimensionalities of the arguments
        
        self._device = device

    def getInfoDict(self) -> dict:
        info_dict = super().getInfoDict()
        
        info_dict['channel_ord'] = self._channel_ord
        info_dict['target_channel'] = self._target_channels
        info_dict['preprocessors_on'] = 0 if self.preprocessors is None else 1
        info_dict['core_idx'] = self._core

        return info_dict
    
    def setPreprocessors(self, preprocessor_cls, preprocessor_kwargs: dict, force_reset: bool = False):
        if force_reset or self.preprocessors is None:
            if force_reset:
                warn(f"Forcefully resetting parameters of NO agent in {id(self)}.")
            print(preprocessor_cls)
            self.preprocessors = []
            self.preprocessors.append(preprocessor_cls(**preprocessor_kwargs)) # setting in-prepocessor
            self.preprocessors.append(preprocessor_cls(**preprocessor_kwargs)) # setting out-prepocessor

    # def trainPrepocessor(self,
    #                      env: NeuralOpSystemEnvironment,
    #                      loader: torch.utils.data.Dataset,
    #                      normalizer_cls,
    #                      norm_kwargs: dict):
          
    #     in_normalzier = normalizer_cls(**norm_kwargs)
    #     out_normalizer = normalizer_cls(**norm_kwargs)

    #     for idx, sample in enumerate(train_dataset):
    #         sample_x = sample['x'].to('cuda')
    #         sample_y = sample['y'].to('cuda')

    #         if (idx % 100) == 0:
    #             print(f'Processing train sample {idx}: shapes are {sample_x.shape, sample_y.shape}, device: {sample_x.device}')
    #         inp_normalizer.partial_fit(sample_x)
    #         out_normalizer.partial_fit(sample_y)

    @property
    def adapters(self): # -> Tuple[torch.nn.Module]
        # TODO: necessary asserts
        return self._adapters
    
    @adapters.setter
    def adapters(self, passed_adapters) -> None:
        # TODO: necessary asserts

        print(f'In adapeters-setter: {passed_adapters}')
        if passed_adapters is None:
            self._adapters = None
        else:
            assert len(passed_adapters) == 2, 'arguments of the adapters setter have to include both lifting and projection'
            self._adapters = passed_adapters
        print(self._adapters)

    #   loading_info_dict: dict, # {'cores': {'p': ..., 's': ...}, 'adapters': {0: (...), ..., N: ()}}
    def loadAgent(self, model_paths: Tuple[str], preprocessors_paths: Tuple[str] = None):
        assert isinstance(model_paths, tuple) and len(model_paths) == 2, \
            'Loading agent projection model requires tuple of str arg with len 3'
        
        adapter_params_0 = torch.load(f = model_paths[0], **self._SAVE_LOAD_PARAMS)[0]
        adapter_params_1 = torch.load(f = model_paths[1], **self._SAVE_LOAD_PARAMS)[0]

        print(f'TYPES OF LOADED ADAPTERS: {type(adapter_params_0)}, {type(adapter_params_1)}')

        if isinstance(adapter_params_0, dict):
            try:
                adapters_in = PostLiftMambaLifting(width = 32,
                                                use_mamba_kwargs = None,
                                                mamba_fallback_kernel = 9,
                                                padding = 0,
                                                n_dim = 3,
                                                non_linearity = torch.nn.functional.gelu)
                adapters_in.load_state_dict(adapter_params_0)
            except: # GET WIDTH OF CORE
                adapters_in = ChannelMLP(in_channels=len(self._channel_ord),
                                        out_channels=32, # Get width of a specific core from self.env._cores
                                        hidden_channels = 32,
                                        n_layers = 2,
                                        n_dim = 3,
                                        non_linearity = torch.nn.functional.gelu)
                adapters_in.load_state_dict(adapter_params_0)
        elif isinstance(adapter_params_0, torch.nn.Module):
            adapters_in = adapter_params_0
        else:
            print(adapter_params_0, type(adapter_params_0))
            raise(ValueError('Incorrect form of loaded operators'))
        
        if isinstance(adapter_params_1, dict):
            adapters_out = ChannelMLP(in_channels=32, # Get width of a specific core from self.env._cores
                                    out_channels=1, # Get number of outputs for a specific operator
                                    hidden_channels = 32,
                                    n_layers = 2,
                                    n_dim = 3,
                                    non_linearity = torch.nn.functional.gelu)
            adapters_out.load_state_dict(adapter_params_1)
        else:
            adapters_out = adapter_params_1

        print(f'--------- Adapters setter is about to be called ---------')
        self._adapters = torch.nn.ModuleList([adapters_in, adapters_out])
        print(f'--------- Adapters setter was called. Was it? ---------')

        print(f'Setting adapter as {self.adapters} from {torch.nn.ModuleList([adapters_in, adapters_out])}')

        if preprocessors_paths is not None:
            assert (len(preprocessors_paths) == 2 and 
                    isinstance(preprocessors_paths[0], str) and 
                    isinstance(preprocessors_paths[1], str)), 'Incorrect inputs for preprocessor loading'
            self.preprocessors[0].from_file(preprocessors_paths[0])
            self.preprocessors[1].from_file(preprocessors_paths[1])

    #   saving_info_dict: dict, # {'cores': {'p': ..., 's': ...}, 'adapters': {0: (...), ..., N: ()}}
    def saveAgent(self, model_paths: Union[List[str], Tuple[str]], preprocessors_paths: Tuple[str] = None) -> Tuple[int, dict]:
        assert isinstance(model_paths, (list, tuple)) and len(model_paths) == 2, \
            'Saving lifting-main part-projection model of an agent requires tuple of str arg with len 2'
        
        torch.save(obj = self.adapters[0], f = model_paths[0], **self._SAVE_LOAD_PARAMS)
        torch.save(obj = self.adapters[1], f = model_paths[1], **self._SAVE_LOAD_PARAMS)        

        if (preprocessors_paths is not None) and (self.preprocessors is not None):
            assert (len(preprocessors_paths) == 2 and 
                    isinstance(preprocessors_paths[0], str) and 
                    isinstance(preprocessors_paths[1], str)), 'Incorrect inputs for preprocessor saving'
            self.preprocessors[0].to_file(preprocessors_paths[0])
            self.preprocessors[1].to_file(preprocessors_paths[1])

        return self._ID, self.getInfoDict()

    def formInput(self) -> torch.Tensor:
        # print(f'Forming input with channels {self._channel_ord}')
        return self.matchDevice(self.env.access(self._channel_ord))

    def matchDevice(self, arg: torch.Tensor):
        if not arg.device == self._device:
            arg.to(self._device)

        return arg

    def initPreprocessors(self, mask: torch.Tensor, dim: List[int] = [2, 3, 4]):
        self.preprocessors = [UnitGaussianNormalizer(dim = dim, mask = mask),
                              UnitGaussianNormalizer(dim = dim, mask = mask)]

    def callCore(self, lifted_x: torch.Tensor) -> torch.Tensor:
        # TODO: necessary asserts
        return self.env.callCore(self._core, lifted_x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: add necessary asserts about tensors
        assert self.adapters is not None, 'Adapters are not yet declared, can not proceed further.'

        if self.preprocessors is not None:
            if isinstance(self.preprocessors, (list, tuple)):
                x = self.preprocessors[0].transform(x)
            else:
                print(type(self.preprocessors))
                print(self.preprocessors)
                raise NotImplementedError('DefaultDataProcessor is too inconvenient, use Transform subclasses')

        x = self.adapters[0](x)
        x = self.callCore(x)
        x = self.adapters[1](x)

        if self.preprocessors is not None:
            if isinstance(self.preprocessors, (list, tuple)):
                x = self.preprocessors[1].inverse_transform(x)
            else:
                raise NotImplementedError('DefaultDataProcessor is too inconvenient, use Transform subclasses')        

        return x

    def passMessage(self, message: Message, *args, **kwargs) -> bool:
        # Tuple[bool, Dict[int, torch.Tensor]]: # Or just bool as a flag of correct execution?
        # pred_ID: int, # inp: Union[torch.Tensor, Dict[int, torch.Tensor], List[torch.Tensor]],

        self.setPredecessorAsExecuted(message.ID)

        if not self.checkReadiness():
            print(f'Operator {self._ID} still misses execution requirements. pred exec dict is {self._predecessors_exec} with {message.ID}.')
            return False
        
        self.env.modifyChannels(self._target_channels, self(self.formInput()))
        
        scs_IDs = list(self._successors.keys()) # [for key in self._successors.keys()]
        message = CalcMessage(self._ID, scs_IDs, self._channel_ord, None)

        for scs_ID, scs_agent in self._successors.items():
            exec_flag = scs_agent.passMessage(message)

        return True


class DataPatchingAgent(BasicAgent):
    def __init__(self, axis: int, window_size: int):
        raise NotImplementedError('Not ready yet.')
        self._axis = axis
        self._window_size = window_size

        self.reset()

    def reset(self) -> None:
        self._cur_pos = 0

    def __call__(self, arg: torch.Tensor) -> torch.Tensor:
         # Maybe use torch split instead and then feed other agents
        arg = torch.narrow(arg, self._axis, self._cur_pos, self._window_size)
        self._cur_pos += self._window_size
        return arg


class FixedMultiAgentSystem(torch.nn.Module):
    '''
    In current iteration, the MAS will be just a pre-defined set of agents with fixed roles and inter-agent links.
    '''
    ID_low  = 1 
    ID_high = 1e2 # Probably, there will be no need in more than 100 agents

    def __init__(self, env: NeuralOpSystemEnvironment, device: str = 'cuda'): # , channel_meta: List[int]
        super().__init__()

        self._unused_IDs = np.arange(self.ID_low, self.ID_high)
        self._agents_torch = torch.nn.ModuleDict()
        self._agents_default = {}
        self._initial_agent_ID = None

        self.env = env

        self.device = device

        # self._channel_meta = channel_meta
        # agents shall be in form ID "int" : (agent "subclass of BasicAgent", list of outgoing agents "list of ID ints")

    def addAgent(self, agent: AbstractAgent, agent_ID: int = None) -> None: # outgoings: List[int] = None, 
        if agent_ID is None:
            valid_IDs = self._unused_IDs[~np.isin(self._unused_IDs, list(self._agents.keys()))]
            agent_ID = np.random.choice(valid_IDs)
        elif agent_ID in self._agents_torch.keys() or agent_ID in self._agents_default.keys():
            raise KeyError(f'Trying to add agent with ID {agent_ID}, even though the ID' + 
                           f' is already used by {self._agents[agent_ID]} in the MAS.')

        # Refined from self._agents[agent_ID] = agent:

        if isinstance(agent, torch.nn.Module): #torch.nn.Module in agent.__mro__:
            self._agents_torch[str(agent_ID)] = agent
        else:
            self._agents_default[agent_ID] = agent
        
    # def linkAgent(self, ID_from: int, ID_to: int):
    #     assert isinstance(self._agents[ID_from][1], list), 'Second position of agent in dict must be a list' 
    #     self._agents[ID_from][1].append(ID_to)

    def forward(self, x):
        # self.env.setTensor(initial_tensor = x, channel_meta = self._channel_meta)
        # assert (0 in self._agents_torch.keys()) or (0 in self._agents_default.keys()), \
        #     'Initial conditions operator is missing from MAS'
        x = x.to(self.device)

        if str(0) in self._agents_torch.keys():
            self._agents_torch[0].passMessage(x)
        elif 0 in self._agents_default.keys():
            self._agents_default[0].passMessage(x)
        else:
            raise ValueError('Initial conditions operator is missing from MAS')

        return self.env.getVarState()
    
    def setPreprocessors(self, preprocessor_cls, preprocessor_kwargs: dict, force_reset: bool = False):
        for agent_ID, agent in self._agents_torch.items():
            if isinstance(agent, NeuralOperatorAgent):
                agent.setPreprocessors(preprocessor_cls, preprocessor_kwargs, force_reset)

def formICTensor(arg: torch.Tensor):
    '''
    Returns initial conditions tensor in shape [B x C x T x X x ...] 
    from the argument with shape [B x T x C x X x ...]
    '''
    shape = list(arg.shape)
    shape[1], shape[2] = shape[2], shape[1]

    return arg[:, 0:1 , ...].permute(0, 2, 1, 3, 4).expand(size = shape)


def formInputForPreprocessor(sample_x: torch.Tensor, sample_y: torch.Tensor,
                             arg_channels: List[int], target_channels: List[int], device: str = 'cuda') -> Tuple[torch.Tensor]:
    '''
    Both samples have to be sent by a batch and, thus, shaped [C x T x X x ...]
    '''
    # prep_input = torch.clone(sample_x)
    channels_in = [] # = {'in': [], 'out': []}

    for arg_chan in arg_channels:
        if arg_chan in target_channels:
            channels_in.append(sample_y[arg_chan:arg_chan+1])
        else:
            channels_in.append(sample_x[arg_chan:arg_chan+1])

    # print('In formInputForPreprocessor', torch.cat(channels_in, dim=0).shape, sample_y[tuple(target_channels)].shape)
    output_in  = torch.cat(channels_in, dim=0).to(device)
    output_out = sample_y[tuple(target_channels)]
    if output_in.ndim != output_out.ndim:
        output_out = torch.unsqueeze(output_out, dim = 0).to(device)

    return output_in, output_out 


def trainMASPreprocessorsOnDataset(mas: FixedMultiAgentSystem, dataset: torch.utils.data.Dataset):
    print('dataset has the type: ', type(dataset))
    for sample_idx, sample in enumerate(dataset):
        sample_x, sample_y = sample['x'].to('cuda'), sample['y'].to('cuda')

        if (sample_idx % 100) == 0:
            print(f'Processing train sample {sample_idx}: shapes are {sample_x.shape, sample_y.shape} on {sample_x.device}')

        # mas.env.setTensor(sample_x)
        for agent in mas._agents_torch.values():
            if isinstance(agent, NeuralOperatorAgent):
                assert agent.preprocessors is not None, 'Trying to fit empty preprocessors.'
                agent_arg_in, agent_arg_out = formInputForPreprocessor(sample_x, sample_y, 
                                                                       agent._channel_ord, agent._target_channels, mas.device)

                # print('Shapes: in - ', agent_arg_in.shape, ' out - ', agent_arg_out.shape, 'device', agent_arg_in.device)
                agent.preprocessors[0].partial_fit(agent_arg_in)
                agent.preprocessors[1].partial_fit(agent_arg_out)


if __name__ == "__main__":
    env = NeuralOpSystemEnvironment()

    tensor_1 = torch.rand(size=(10, 1, 12, 32, 32))
    tensor_2 = torch.rand(size=(10, 2, 12, 32, 32))
    tensor_3 = torch.rand(size=(10, 1, 12, 32, 32))

    initial_cond_loader = InitialConditionsAgent(env = env,
                                                 initial_conds = {'1': tensor_1,
                                                                  '2': tensor_2,
                                                                  '3': tensor_3},
                                                 additional_channels = torch.rand(size=(10, 3, 12, 32, 32)), 
                                                 )
    
    adapters_p, adapters_sw, adapters_so = (), (), ()
    core_p, core_s = None, None

    pressure_agent = NeuralOperatorAgent(ID = 1,
                                         channel_ord=[0,],
                                         env = env,
                                         adapters = adapters_p,
                                         core = core_p,
                                         predecessors=[(0, initial_cond_loader)])
    
    saturation_water_agent = NeuralOperatorAgent(ID = 2,
                                                 channel_ord=[1,],
                                                 env = env,
                                                 adapters = adapters_sw,
                                                 core = core_s,
                                                 predecessors=[(1, pressure_agent),])

    saturation_oil_agent = NeuralOperatorAgent(ID = 3,
                                               channel_ord=[2,],
                                               env = env,
                                               adapters = adapters_so,
                                               core = core_s,
                                               predecessors=[(1, pressure_agent),])

    saturation_gas_agent = NeuralOperatorAgent(ID = 4,
                                               channel_ord=[2,],
                                               env = env,
                                               adapters = adapters_so,
                                               core = core_s,
                                               predecessors=[(1, pressure_agent),])

    mas = FixedMultiAgentSystem(env=env)

    mas.addAgent(initial_cond_loader, 0)

    mas.addAgent(pressure_agent, 1)

    mas.addAgent(saturation_water_agent, 2)
    mas.addAgent(saturation_oil_agent, 3)
    mas.addAgent(saturation_gas_agent, 4)

    mas.forward()

    # pressure_agent.addSuccessors([(1, saturation_water_agent), (2, saturation_oil_agent)])
    # initial_cond_loader.passMessage()
    # pred = env.getVarState()