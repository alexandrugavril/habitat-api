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
import numpy as np

import habitat
from habitat import Config, Dataset, SimulatorActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


@baseline_registry.register_env(name="NavObjectRLEnv")
class NavObjectRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._core_env_config = config.TASK_CONFIG
        self._rl_config = config.RL

        task_cfg = self._core_env_config.TASK

        self._success_if_in_view = task_cfg.SUCCESS_IF_IN_VIEW
        self._view_field = task_cfg.VIEW_FIELD_FACTOR
        self._with_collision_reward = self._rl_config.COLLISION_REWARD_ENABLED
        self._collision_reward = self._rl_config.COLLISION_REWARD
        self._collision_distance = self._rl_config.COLLISION_DISTANCE
        self._num_steps = 0
        self._min_start_dist_to_goal = 1.
        super().__init__(config, dataset)

    def reset(self):
        self._previous_action = None
        #
        # if self._env._current_episode is not None:
        #     ep = self._env.current_episode
        #     goal_idx = ep.goal_idx
        #     success_distance = self._success_distance + ep.goals[goal_idx].radius
        #
        #     print("finished: ", ep.scene_id, self._num_steps,
        #           self._distance_target(), \
        #                          success_distance)
        #
        self._num_steps = 0

        dist_to_goal = 0
        min_dist = self._min_start_dist_to_goal
        th_dist = self._success_distance
        observations = super().reset()

        while dist_to_goal < min_dist:
            observations = super().reset()
            ep = self._env.current_episode

            print(ep)
            input()
            continue

            success_distance = th_dist + ep.goals[ep.goal_idx].radius
            dist_to_goal = self._distance_target() - success_distance

        goal_idx = self._env.current_episode.goal_idx
        # print(self._env.current_episode.scene_id,
        #       self._env.current_episode.episode_id)

        self._previous_target_distance = \
            self.habitat_env.current_episode.geodesic_distances[goal_idx]
        return observations

    def step(self, *args, **kwargs):
        self._num_steps += 1
        self._previous_action = kwargs["action"]
        observation, reward, done, info = super().step(*args, **kwargs)

        if self._with_collision_reward:
            if self._collision_distance <= 0:
                if info["collisions"]["is_collision"]:
                    reward += self._collision_reward
            else:
                if observation["proximity"][0] < self._collision_distance:
                    reward += self._collision_reward

        # print(self._env.current_episode.scene_id,
        #       self._env.current_episode.episode_id, "NUMSTEP:",
        #       self._num_steps)
        # ep = self._env.current_episode
        # goal_idx = ep.goal_idx

        # print("STEP INFO:", self._env.current_episode.goal_idx,
        #       self._distance_target(), reward, done, ep.goals[goal_idx].radius)
        return observation, reward, done, info
        
    def _distance_target(self):
        goal_idx = self._env.current_episode.goal_idx

        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[goal_idx].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        # return False
        ep = self._env.current_episode
        goal_idx = ep.goal_idx
        success_distance = self._success_distance + ep.goals[goal_idx].radius
        if (
            # self._env.task.is_stop_called
            # and
            self._distance_target() < success_distance
        ):
            if self._success_if_in_view:
                goal_stats = ep.goal_coord_in_camera
                _, _, _, xpx, ypx = goal_stats
                view_d = np.linalg.norm([xpx, ypx])
                if view_d < self._view_field:
                    return True
            else:
                return True

        return False


@baseline_registry.register_env(name="NavObjectClassRLEnv")
class NavObjectClassRLEnv(NavObjectRLEnv):
    def _episode_success(self):
        ep = self._env.current_episode
        goal_idx = ep.goal_idx
        success_distance = self._success_distance + ep.goals[goal_idx].radius

        if (
            self._env.task.is_stop_called
            and self._distance_target() < success_distance
        ):
            return True
        return False
