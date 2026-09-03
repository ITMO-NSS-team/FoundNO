import logging
import os
from functools import singledispatchmethod
from typing import Dict

import numpy as np

LOGFORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def formLog(arg: Dict[str, list]):
    return " | ".join([key + " " + str(np.mean(val)) for key, val in arg.items()])


class Logger:
    def __init__(
            self,
            filename,
            log_level=logging.INFO,
            logger_name: str = "FoundationalFNO",
            write_every: int = 1,
            epochs_aggreg: int = 1,
            info_entries: list = None,
    ):
        if info_entries is None:
            info_entries = []

        self._filename = filename
        self._log_level = log_level
        self._logger_name = logger_name

        self._idx_internal = 0
        self._info_entries = info_entries
        self._info_dict = {entry: [] for entry in info_entries}

        self._write_every = int(write_every)
        assert epochs_aggreg > 0, "epochs_aggreg must have positive value"
        self._aggreg_count = int(epochs_aggreg)

        self.setLogger()

    def getInfoDict(self):
        return {entry: [] for entry in self._info_entries}

    def setLogger(self):
        log_dir = os.path.dirname(self._filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if self._logger_name is not None:
            self._logger = logging.getLogger(self._logger_name)
        else:
            self._logger = logging.getLogger()

        self._logger.setLevel(self._log_level)
        self._logger.propagate = False

        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter(LOGFORMAT)

        file_handler = logging.FileHandler(self._filename, encoding="utf-8")
        file_handler.setLevel(self._log_level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self._log_level)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    @singledispatchmethod
    def write(self, arg):
        raise NotImplementedError(
            f"Can not call write into log method for {type(arg)} objects."
        )

    @write.register
    def _(self, arg: dict):
        if self._info_entries:
            assert set(arg.keys()) == set(self._info_dict.keys()), (
                "Passed logs do not match required entries."
            )

            if (self._idx_internal % self._write_every) < self._aggreg_count:
                for key, value in arg.items():
                    self._info_dict[key].append(value)

            if (self._idx_internal % self._write_every) == self._aggreg_count - 1:
                self._logger.info(
                    "Epoch {} | ".format(
                        self._idx_internal + 1 - self._aggreg_count
                    )
                    + formLog(self._info_dict)
                )
                self._info_dict = self.getInfoDict()

            self._idx_internal += 1
        else:
            self._logger.info(
                " | ".join(f"{key} {value}" for key, value in arg.items())
            )

    @write.register
    def _(self, arg: str):
        self._logger.info(arg)
