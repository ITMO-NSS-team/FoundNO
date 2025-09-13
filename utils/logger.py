import os
import logging

LOGFORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

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
    def __init__(self, filename, log_level = logging.INFO, logger_name: str = 'FoundationalFNO'):
        logging.basicConfig(format=LOGFORMAT, level=log_level)
        self._filename = filename
        self._log_level = log_level
        self._logger_name = logger_name
        self.set_logger()

    def set_logger(self):
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

    def write(self, arg: str):
        self._logger.info(arg)