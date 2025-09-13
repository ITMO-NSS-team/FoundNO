import os, sys
import logging
import warnings
from typing import List, Dict

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from galore_torch import GaLoreAdamW
    GALORE_FLAG = 1
except ImportError:
    warnings.warn('GaLore optimizer is unavailable due to missing lib.')
    GALORE_FLAG = 0

def set_scheduler(args: dict, optimizer : torch.optim.Optimizer):
    """ set the lr scheduler """
    if args['scheduler'] == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args['patience'], 
                                                   verbose=True, min_lr=1e-3*1e-5, factor=0.2)
    elif args['scheduler'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['max_cosine_lr_epochs'])
    else:
        scheduler = None
    return scheduler

def set_optimizer(args: dict, parameters: List[dict]):
    """ 
    Set the optimizer for both FNO part and lifting layers.

    parameters (List[dict]): parameters of FNO network and lifting networks for all passed equation liftings.

    Examples:
    >>> set_optimizer(args, {'params': fno.parameters()}, {'params': lifting1.parameters()}, {'params': lifting2.parameters()}, ...)

    """
    # To consider: make higher learning rates for parameters of "expert" neural networks
    if args['optimizer'] == "adam":
        optimizer = optim.Adam(parameters, lr=args['lr'])
    elif args['optimizer'] == "sgd":
        optimizer = optim.SGD(parameters, lr=args['lr'], momentum=0.9)
    elif args['optimizer'] == "galore_adamw":
        optimizer = GaLoreAdamW()
    return optimizer