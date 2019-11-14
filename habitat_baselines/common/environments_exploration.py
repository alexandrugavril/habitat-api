#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Any, Dict, Optional, Type, Union

import habitat
from habitat import Config, Dataset, SimulatorActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.augmentation_env import AugmentEnv

import numpy as np
import torch


@baseline_registry.register_env(name="NavRLExploration")
class NavRLExplorationEnv(AugmentEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._slack_reward = config.RL.SLACK_REWARD
        self._success_reward = config.RL.SUCCESS_REWARD

        self._previous_action = None
        self._grid_resolution = self._rl_config.REACHABILITY.grid_resolution

        self._collected_positions = set()
        super().__init__(config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._collected_positions = set()

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        observation, reward, done, info = super().step(*args, **kwargs)

        return observation, reward, done, info

    def get_reward_range(self):
        return (
            0,
            1,
        )

    def get_reward(self, observations):
        agent_position = self._env.sim.get_agent_state().position.tolist()
        x, y, z = agent_position

        reward = 0

        quantized_x = int(x / self._grid_resolution)
        quantized_y = int(y / self._grid_resolution)
        quantized_z = int(z / self._grid_resolution)
        position_id = (quantized_x, quantized_y, quantized_z)

        if position_id not in self._collected_positions:
            reward += self._success_reward
            self._collected_positions.add(position_id)
        else:
            reward += self._slack_reward

        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
