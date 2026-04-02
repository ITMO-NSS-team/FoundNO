import sys
import os
import pickle

from typing import List, Dict, Tuple, Union
from warnings import warn

import torch

from fnofound.models.fno import FNO
from fnofound.models.local_no import LocalNO

from fnofound.models.coda import CODANO
from fnofound.models.pecoda import PeCODANO
from fnofound.models.mamba_fno import PostLiftMambaFNO3D, PostLiftMambaLifting
from fnofound.models.localattn_exp import LocalAttnFNO
from neuralop.layers.channel_mlp import ChannelMLP

from fnofound.agents.agent import (NeuralOpSystemEnvironment, FixedMultiAgentSystem,
                                   AbstractAgent, BasicAgent, NeuralOperatorAgent, InitialConditionsAgent) 


def initAgent(ID: int, key: str, env: NeuralOpSystemEnvironment, **agent_kwargs) -> AbstractAgent:
    match key:
        case 'b':
            warn('An agent was set to be initialized as a BasicAgent, which is pointless.')
            return BasicAgent(ID = ID, env = env, **agent_kwargs)
        case 'a':
            raise RuntimeError('Trying to initialize abstract agent.')
        case 'i':
            return InitialConditionsAgent(ID = ID, env = env, **agent_kwargs)
        case 'n':
            return NeuralOperatorAgent(ID = ID, env = env, **agent_kwargs)


def loadMultiAgentSystem(env: NeuralOpSystemEnvironment,
                         info_pkl_filename: str,
                         models: Dict[int, Tuple[str]],
                         preprocessors: Dict[int, Tuple[str]] = None) -> FixedMultiAgentSystem:
    # info dict will be smth like {'operators': {},
    #                              'adapters': {0: (...), ..., N: ()},
    #                              'preprocessors': {0: (...), ..., N: ()}}

    if preprocessors is None:
        preprocessors = {}

    with open(info_pkl_filename, 'wb') as pkl_file:  
        mas_info = pickle.load(file = pkl_file)

    agents = {}
    for agent_ID, agent_properties in mas_info:
        init_agent_info = {}
        if agent_properties['meta'][0] != 'i':
            init_agent_info['pred_agents'] = [(agent_id, agents[agent_id]) for agent_id in agent_properties['predecessors']]
        else:
            init_agent_info['pred_agents'] = None

        init_agent_info['successors'] = None

        if agent_properties['meta'][0] == 'n':
            init_agent_info['adapters'] = None
            init_agent_info['core'] = None
        
        new_agent = initAgent(ID = agent_properties['ID'], key = agent_properties['meta'][0], env = env,
                              **init_agent_info)
        
        if agent_properties['meta'][0] == 'n':
            cur_preprocessor = preprocessors[agent_ID] if agent_ID in preprocessors.keys() else None

            new_agent.loadAgent(models[agent_ID], cur_preprocessor)

        for pred_agent_ID in agent_properties['predecessors']:
            agents[pred_agent_ID].addSuccessors((agent_properties['ID'], new_agent))
        agents[agent_ID] = new_agent

    mas = FixedMultiAgentSystem(env)
    for agent_id, agent in agents.items():
       mas.addAgent(agent, agent_id) 
    return mas


def saveMultiAgentSystem(mas: FixedMultiAgentSystem, #Union[Dict[int, AbstractAgent], FixedMultiAgentSystem],
                         filenames: Dict[str, Union[Tuple[str], Dict[str, Tuple[Tuple[str]]]]],
                         env: NeuralOpSystemEnvironment = None):
    if env is None and isinstance(mas, dict):
        warn('No environment to save, bypassing the step')
    elif env is not None:
        assert 'cores' in filenames.keys(), 'No provided filenames for the cores.'
        env.saveEnv(filenames['cores'][0], filenames['cores'][1])
    else:
        assert isinstance(mas, FixedMultiAgentSystem), 'Assert to avoid undefined behavior: agents have to be in a dict, or fixed MAS'
        mas.env.saveEnv(filenames['cores'][0], filenames['cores'][1])

    to_save = {}
    if isinstance(mas, FixedMultiAgentSystem):
        # TODO: rework loop into a set of 
        for agent_id, agent in mas._agents_default.items():
            # assert agent_id in filenames['agents'].keys(), f'ID {agent_id} is missing from filenames.'
            if agent_id in filenames['agents'].keys():
                if not (isinstance(filenames['agents'][agent_id], (list, tuple))    and 
                        len(filenames['agents'][agent_id])                          and 
                        isinstance(filenames['agents'][agent_id][0], (list, tuple)) and
                        all([isinstance(elem, str) for elem in filenames['agents'][agent_id][0]])):
                    raise ValueError(f'Incorrect path passed to save the MAS agent with ID {agent_id}')

                preprocessing_paths = filenames['agents'][agent_id][1] if len(filenames['agents'][agent_id]) == 2 else None
                print(f"Saving for {filenames['agents'][agent_id]}")
                cur_ID, info = agent.saveAgent(model_paths = filenames['agents'][agent_id][0], 
                                               preprocessors_paths = preprocessing_paths)# filenames['agents'][agent_id][1])
                assert cur_ID == agent_id, 'Agent indexation went wrong.'
                to_save[cur_ID] = info
            else:
                to_save[agent_id] = agent.getInfoDict()


        for agent_id, agent in mas._agents_torch.items():
            assert int(agent_id) in filenames['agents'].keys(), f'ID {int(agent_id)} is missing from filenames {filenames["agents"].keys()}.'
            if not (isinstance(filenames['agents'][int(agent_id)], (list, tuple))    and 
                    len(filenames['agents'][int(agent_id)])                          and 
                    isinstance(filenames['agents'][int(agent_id)][0], (list, tuple)) and
                    all([isinstance(elem, str) for elem in filenames['agents'][int(agent_id)][0]])):
                raise ValueError(f'Incorrect path passed to save the MAS agent with ID {int(agent_id)}')
            
            preprocessing_paths = filenames['agents'][int(agent_id)][1] if len(filenames['agents'][int(agent_id)]) == 2 else None
            cur_ID, info = agent.saveAgent(model_paths = filenames['agents'][int(agent_id)][0], 
                                           preprocessors_paths = preprocessing_paths)
            assert int(cur_ID) == int(agent_id), 'Agent indexation went wrong.'
            assert int(cur_ID) not in to_save.keys(), f'Key {cur_ID} already in to_save dict'
            to_save[int(cur_ID)] = info

    else:
        raise NotImplementedError("Incorrect type")

    with open(filenames['info'], 'wb') as pkl_file:  
        pickle.dump(obj = to_save, file = pkl_file)

    # if isinstance(agents, FixedMultiAgentSystem):
    #     agents = agents._agents

    # for agent_id, agent in agents.items():
    #     assert agent_id in filenames['agents'].keys(), f'ID {agent_id} is missing from filenames.'
    #     preprocessing_paths = filenames['agents'][agent_id][1] if len(filenames['agents'][agent_id]) == 2 else None
    #     agent.saveAgent(filenames['agents'][agent_id][0], preprocessing_paths)