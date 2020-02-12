#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
import torch
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import gym
import numpy as np
from gym.spaces import Dict as SpaceDict

import habitat
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images
from habitat.core.vector_env import VectorEnv

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"


def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    """Constructor for default habitat `env.Env`.

    :param config: configuration for environment.
    :param dataset: dataset for environment.
    :param rank: rank for setting seed of environment
    :return: `env.Env` / `env.RLEnv` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    return habitat_env


class VectorEnvSharedMem(VectorEnv):

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        shared_mem: List[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            make_env_fn, env_fn_args, auto_reset_done,
            multiprocessing_start_method
        )

        mem_obs = ["rgb", "depth"]
        self.shared_mem = dict()
        for i in range(len(shared_mem)):
            self.shared_mem[mem_obs[i]] = shared_mem[i]

    def get_shared_mem(self):
        return self.shared_mem



