import logging
import os
from collections import OrderedDict

import yaml


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_options(root_path):
    # parse yml to dict
    with open(root_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    return opt

def setup_logger(logger_name='base',
                 logger_file='log.log',
                 level=logging.INFO,
                 screen=False,
                 tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        if os.path.exists(logger_file):
            mode='a+'
        else:
            mode = 'w'
        fh = logging.FileHandler(logger_file, mode=mode)
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)