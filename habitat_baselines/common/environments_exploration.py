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
        self._core_env_config = task = config.TASK_CONFIG
        self._slack_reward = config.RL.SLACK_REWARD
        self._success_reward = config.RL.SUCCESS_REWARD
        assert self._success_reward > 0, "Must have SUCCESS_REWARD > 0"

        self._previous_action = None
        self._grid_resolution = self._rl_config.REACHABILITY.grid_resolution

        # -- Collision
        self._with_collision_reward = self._rl_config.COLLISION_REWARD_ENABLED
        self._collision_reward = self._rl_config.COLLISION_REWARD
        self._collision_distance = self._rl_config.COLLISION_DISTANCE

        self._no_op = self._rl_config.NO_OPERATION

        width = task.SIMULATOR.DEPTH_SENSOR.WIDTH
        block_w = int(self._rl_config.DEPTH_BLOCK_VIEW_FACTOR * width)
        self._depth_lim = [width//2 - block_w, width//2 + block_w]
        self._depth_min_unblock = self._rl_config.DEPTH_BLOCK_MIN_UNBLOCK

        self._prev_collision = False

        self._collected_positions = set()
        super().__init__(config, dataset)

    def reset(self):
        self._previous_action = None
        self._prev_collision = False

        observations = super().reset()

        self._collected_positions = set()

        return observations

    def unblocked(self, action):
        lim = self._depth_lim
        min_depth = self._depth_min_unblock
        obs = self._env.sim.get_observations_at()
        obs.update(
            self._env.task.sensor_suite.get_observations(
                observations=obs,
                episode=self._env.current_episode,
                action=action,
                task=self._env.task,
            )
        )

        depth = obs["depth"]

        zone = depth[:, lim[0]: lim[1]]

        if zone.min() > min_depth:
            return True, None

        super_get_obs = getattr(super(), "_process_obs", None)
        if super_get_obs is not None:
            obs = super_get_obs(obs)

        return False, obs

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]

        skip_new_obs = False

        # TODO fix hardcoded 0 action <-> forward
        act = kwargs["action"]["action"]

        if self._no_op and self._prev_collision and act == 0:
            raise NotImplemented
            unlock, observation = self.unblocked(kwargs["action"])

            if not unlock:
                # TODO might be an augmentated collision
                reward = self.get_reward(observation)
                done = self.get_done(observation)
                info = self.get_info(observation)
                skip_new_obs = True
            else:
                self._prev_collision = False

        if not skip_new_obs:
            observation, reward, done, info = super().step(*args, **kwargs)

        # Add slack and register discovered score
        info["discovered"] = reward > 0
        info["collisions_wall"] = 0
        info["collisions_prox"] = 0

        reward += self._slack_reward

        reward += self._slack_reward

        # Do not have proximity in sensors but in info
        if self._with_collision_reward:
            collision = False
            if self._collision_distance <= 0:
                if info["collisions"]["is_collision"]:
                    collision = True
                    info["collisions_wall"] = 1
            else:
                if observation["proximity"][0] < self._collision_distance:
                    collision = True
                    info["collisions_prox"] = 1

            if collision:
                reward += self._collision_reward
                self._prev_collision = True

        return observation, reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
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
        # else:
        #     reward += self._slack_reward

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
