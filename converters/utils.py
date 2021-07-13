#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains code for utilities
_____________________________________________________________________________
"""

from pathlib import Path
import logging
from collections import OrderedDict
from config import _C

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def initial_logger():
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def experiment_loader(model_format='pth', data_path='./weight'):
    data_path = Path(data_path)
    if not data_path.exists():
        raise Exception("No experiment folder for", data_path)
    if model_format=='pth':
        saved_model = sorted(data_path.glob('*.pth'))
    elif model_format=='onnx':
        saved_model = sorted(data_path.glob('*.onnx'))
    elif model_format=='pt':
        saved_model = sorted(data_path.glob('*.pt'))
    elif model_format=='trt':
        saved_model = sorted(data_path.glob('*.trt'))
    saved_config = sorted(data_path.glob('*.yaml'))

    if len(saved_model) < 1:
        raise Exception("No model format ", model_format, type, "in", data_path)
    if len(saved_config) < 1:
        raise Exception("No config for model format ", model_format, type, "in", data_path)

    return str(saved_model[0]), str(saved_config[0])


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict