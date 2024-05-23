import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import copy
import random
from typing import Dict

import numpy as np
import torch
from train import main as train

from configs.base import Config
from utils.configs import get_options

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_possible_combination_with_key(candidates_dict: Dict):
    from itertools import product

    all_combination = list(product(*candidates_dict.values()))
    all_combination_with_key = []
    for combination in all_combination:
        combination_with_key = {}
        for key, value in zip(candidates_dict.keys(), combination):
            combination_with_key[key] = value
        all_combination_with_key.append(combination_with_key)
    return all_combination_with_key


def update_config(cfg: Config, attr: str, value):
    setattr(cfg, attr, value)
    return cfg


def main(opt_list: Config):
    optim_attributes = opt_list.optim_attributes
    candidates_dict = {}

    for attr in optim_attributes:
        try:
            candidates_dict[attr] = getattr(opt_list, attr)
        except AttributeError:
            raise ValueError(f"Attribute {attr} is not in {opt_list}")

    all_combination_with_key = all_possible_combination_with_key(candidates_dict)

    for combination in all_combination_with_key:
        cfg = copy.copy(opt_list)
        cfg.name += "/"
        for key, value in combination.items():
            cfg = update_config(cfg, key, value)
            cfg.name += f"{key}_{value}|"
        logging.info(f"Start training {cfg.name}")
        train(cfg)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg = get_options(args.config)
    main(cfg)
