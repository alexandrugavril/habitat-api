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

import numpy as np
import torch


@baseline_registry.register_env(name="NavRLExploration")
class NavRLExplorationEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        SimulatorActions.extend_action_space("NOISY_MOVE_FORWARD")
        SimulatorActions.extend_action_space("NOISY_TURN_LEFT")
        SimulatorActions.extend_action_space("NOISY_TURN_RIGHT")

        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._slack_reward = config.RL.SLACK_REWARD
        self._success_reward = config.RL.SUCCESS_REWARD

        self._previous_action = None
        self._grid_resolution = self._rl_config.REACHABILITY.grid_resolution
        self._with_collision_reward = self._rl_config.COLLISION_REWARD_ENABLED
        self._collision_reward = self._rl_config.COLLISION_REWARD
        self._collision_distance = self._rl_config.COLLISION_DISTANCE
        self._eval_mode = config.EVAL_MODE

        self._collected_positions = set()

        self.std_noise = 0.2
        self.depth_min_th = 0.04
        self.depth_max_th = 0.8
        self.noise_interval = 0.03

        super().__init__(self._core_env_config, dataset)

    def process_obs(self, obs):
        if self._eval_mode:
            return obs

        std_noise = self.std_noise
        noise_interval = self.noise_interval

        depth = obs["depth"]

        zone = np.random.rand()
        depth *= ((depth < (zone - noise_interval)) |
                  (depth > (zone + noise_interval)))

        depth_noise = torch.normal(1, std_noise, depth.shape,
                                   device=depth.device)
        depth *= depth_noise
        depth.clamp_(0., 1.)

        depth *= ((depth > self.depth_min_th) & (depth < self.depth_max_th))

        obs["depth"] = depth
        return obs

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        observations = self.process_obs(observations)

        self._collected_positions = set()

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        observation, reward, done, info = super().step(*args, **kwargs)

        observation = self.process_obs(observation)

        if self._with_collision_reward:
            if self._collision_distance <= 0:
                if info["collisions"]["is_collision"]:
                    reward += self._collision_reward
            else:
                if observation["proximity"][0] < self._collision_distance:
                    reward += self._collision_reward

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
