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
        super().__init__(config, dataset)

        task_cfg = self._core_env_config.TASK

        self._success_if_in_view = task_cfg.SUCCESS_IF_IN_VIEW
        self._view_field = task_cfg.VIEW_FIELD_FACTOR

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        goal_idx = self._env.current_episode.goal_idx

        self._previous_target_distance = \
            self.habitat_env.current_episode.geodesic_distances[goal_idx]

        return observations

    def _distance_target(self):
        goal_idx = self._env.current_episode.goal_idx

        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[goal_idx].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        ep = self._env.current_episode
        goal_idx = ep.goal_idx
        success_distance = self._success_distance + ep.goals[goal_idx].radius
        if (
            self._env.task.is_stop_called
            and self._distance_target() < success_distance
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
