import os
import logging
from typing import Dict

from functools import singledispatchmethod

import numpy as np

LOGFORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'

def formLog(arg: Dict[str, list]):
    return ' | '.join([key + ' ' + str(np.mean(val)) for key, val in arg.items()])

class LoggerMeta(type):
    _logger_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        if cls not in cls._logger_instances:
            instance = super().__call__(*args, **kwargs)
            cls._logger_instances[cls] = instance
            
        return cls._logger_instances[cls]
    
    def reset(self):
        self._logger_instances = {}

class Logger(metaclass = LoggerMeta):
    def __init__(self, filename, log_level = logging.INFO, logger_name: str = 'FoundationalFNO',
                 write_every: int = 1, epochs_aggreg: int = 1, info_entries: list = []):
        logging.basicConfig(format=LOGFORMAT, level=log_level)
        self._filename = filename
        self._log_level = log_level
        self._logger_name = logger_name
        self.setLogger()
        
        self._idx_internal = 0
        self._aggregating_mode = True
        self._info_entries = info_entries
        self._info_dict = {entry: [] for entry in info_entries}

        self._write_every = write_every
        assert epochs_aggreg > 0, 'epochs_aggreg must have positive value'
        self._aggreg_count = epochs_aggreg

    def getInfoDict(self):
        return {entry: [] for entry in self._info_entries}

    def setLogger(self):
        if not os.path.exists(os.path.dirname(self._filename)):
            os.makedirs(os.path.dirname(self._filename))

        if self._logger_name is not None:
            self._logger = logging.getLogger(self._logger_name)
        else:
            self._logger = logging.getLogger()

        fh = logging.FileHandler(self._filename)
        fh.setLevel(self._log_level)
        fh.setFormatter(logging.Formatter(LOGFORMAT))
        self._logger.addHandler(fh)

    @singledispatchmethod
    def write(self, arg):
        raise NotImplementedError(f'Can not call write into log method for {type(arg)} objects.')

    @write.register
    def _(self, arg: dict):
        assert arg.keys() == self._info_dict.keys(), 'Passed logs do not match required entries.'
        if (self._idx_internal % self._write_every) < self._aggreg_count:
            for key, value in arg.items():
                self._info_dict[key].append(value)

        if (self._idx_internal % self._write_every) == self._aggreg_count - 1:            
            self._logger.info('Epoch {} | '.format(self._idx_internal +1 - self._aggreg_count) + formLog(self._info_dict))
            self._info_dict = self.getInfoDict()
        self._idx_internal += 1

    @write.register
    def _(self, arg: str):
        self._logger.info(arg)