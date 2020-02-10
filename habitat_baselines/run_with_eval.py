#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import subprocess
import datetime
from typing import List
import yaml
import os

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

RESULTS_FOLDER_PATH = "CHECKPOINT_FOLDER"

# Config patterns
CFG_RESULTS_PREFIX = "results_prefix"
CFG_COMMIT_HASH = "commit_hash"
CFG_GPU = "gpu_id"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        required=False,
        help="GPU ID to use for bot training and test",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        required=False,
        help="Experiment name",
    )
    parser.add_argument(
        "--other-patterns",
        type=str,
        required=False,
        nargs=argparse.ONE_OR_MORE,
        help="Other config regex patterns (list of key value patterns)",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def get_git_revision_hash() -> str:
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD'])).strip()


def preprocess_config(file_content: List[str], change_contet: dict) -> \
    List[str]:
    """
        Change config regex match of change_content key with value

    Returns:
        New file content
    """
    new_file = []
    for line in file_content:
        for k, v in change_contet.items():
            line = line.replace(f"{{{k}}}", f"{v}")
        new_file.append(line)

    return new_file


def run_exp(exp_config: str, run_type: str, opts=None,
            experiment: str = None, gpu_id: int = 0, other_patterns=None) \
    -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if run_type == "train":
        replace_config = dict()

        # Change config based on arguments
        commit_hash = get_git_revision_hash()
        replace_config[CFG_COMMIT_HASH] = commit_hash

        # New folder path based on timestamp & prefix
        fld_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if experiment is not None:
            fld_name += f"_{experiment}"

        replace_config[CFG_RESULTS_PREFIX] = fld_name
        replace_config[CFG_GPU] = gpu_id

        # Add other regex patterns to change in config
        if other_patterns is not None:
            assert len(other_patterns) % 2 == 0, "Must list pairs of arguments"
            for k, v in zip(other_patterns[::2], other_patterns[1::2]):
                replace_config[k] = v
        with open(exp_config, "r") as f:
            config_file_lines = f.readlines()

        config_file_lines = preprocess_config(config_file_lines,
                                              replace_config)

        path = None
        # Get out folder path
        for line in config_file_lines:
            if line.startswith(RESULTS_FOLDER_PATH):
                path = yaml.load(line)[RESULTS_FOLDER_PATH]
                path = path[:path.find(fld_name) + len(fld_name)]
                break

        assert path is not None, "Results path not found"

        # Generate folder for results
        os.makedirs(path)
        cfg_name = os.path.basename(exp_config)
        new_cfg = os.path.join(path, cfg_name)
        with open(new_cfg, "w") as f:
            f.writelines(config_file_lines)

        # Read new generated config
        exp_config = new_cfg

    config = get_config(exp_config, opts)

    # Random seed
    random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    torch.backends.cudnn.deterministic = True  # Slower that normal
    torch.backends.cudnn.benchmark = False

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()

