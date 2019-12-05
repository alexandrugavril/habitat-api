#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.embodied_task import EmbodiedTask, Measure, SimulatorAction
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.tasks.utils import (
    quaternion_from_coeff,
)
import quaternion

from habitat.tasks.nav.nav_task import PointGoalSensor, SPL
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)


CLASSES = dict({  # REPLICA class name: COCO class name
    "book": "book",
    "chair": "chair",
    "table": "diningtable",
    "bowl": "bowl",
    "bottle": "bottle",
    "indoor-plant": "pottedplant",
    "cup": "cup",
    "vase": "vase",
    "tv-screen": "tvmonitor",
    "sofa": "sofa",
    "bike": "bicycle",
    "sink": "sink"
})


@registry.register_sensor
class GoalClass(Sensor):
    r"""
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self.classes = list(CLASSES.keys())
        self.no_classes = len(self.classes)
        self._dimensionality = self.no_classes

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "goalclass"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NORMAL

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=0,
            high=100,
            shape=sensor_shape,
            dtype=np.int,
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):

        class_id = self.classes.index(episode.class_name)
        class_onehot = np.zeros(self.no_classes, dtype=np.int)
        class_onehot[class_id] = 1

        return class_onehot


@registry.register_sensor
class SetGoal(Sensor):
    r"""
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        goal_selection = getattr(config, "GOAL_SELECTION")

        assert goal_selection in self._select_types
        self._goal_selection = self._select_types.index(goal_selection)
        self._prev_goal_idx = None
        self._dimensionality = 1
        self._sim = sim

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "setgoal"

    @property
    def _select_types(self):
        return ["RANDOM", "SAME_RANDOM", "MIN_DIST"]

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NORMAL

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=0,
            high=100,
            shape=sensor_shape,
            dtype=np.int,
        )

    def _select_goal_idx(self, episode, reset):
        goal_selection = self._goal_selection

        # Generate new goal idx only on reset for "RANDOM", "SAME_RANDOM"
        if reset and goal_selection != 2:
            if goal_selection == 0:
                non_inf = np.where(~np.isinf(episode.geodesic_distances))[0]
                goal_idx = np.random.choice(non_inf)
            elif goal_selection == 1:
                non_inf = np.where(~np.isinf(episode.geodesic_distances))[0]
                random_state = np.random.RandomState(episode.episode_id)
                goal_idx = random_state.choice(non_inf)
            self._prev_goal_idx = goal_idx
        else:
            goal_idx = self._prev_goal_idx

        # Check closest goal to determine goal idx
        if goal_selection == 2:
            sim = self._sim
            crt_pos = sim.get_agent_state().position.tolist()
            goals = episode.goals

            dst = np.zeros(len(goals))
            for goal_idx in range(len(goals)):
                dst[goal_idx] = sim.geodesic_distance(
                    crt_pos, goals[goal_idx].position
                )
                goal_idx = np.argmin(dst)

        return goal_idx

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):

        # TODO - Kind of weird - reset goal when no action sent ...
        goal_idx = self._select_goal_idx(episode, "action" not in kwargs)
        goal_idx = np.array([goal_idx], dtype=np.int)
        episode.goal_idx = goal_idx[0]

        return goal_idx


@registry.register_sensor
class AgentPosSensor(Sensor):
    r"""
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = 6

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_pos_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):

        camera_state = self._sim.get_agent_state().sensor_states["rgb"]

        camera_position = camera_state.position
        rotation_world_camera = quaternion.as_float_array(camera_state.rotation)

        # print(type(camera_position))
        # print(camera_position)
        # exit()
        # position = np.array(camera_position,
        #                     dtype=np.float32)
        # rotation_world_camera = np.array(rotation_world_camera,
        #                     dtype=np.float32)
        # print(camera_position.shape)
        # print(rotation_world_camera.shape)
        # exit()
        return np.concatenate([camera_position, rotation_world_camera])


@registry.register_measure
class SPLMultiGoal(SPL):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._success_if_in_view = getattr(config, "SUCCESS_IF_IN_VIEW", True)
        self._view_field = getattr(config, "VIEW_FIELD_FACTOR", 0.2)

        super().__init__(sim=sim, config=config)

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        goal_idx = episode.goal_idx

        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.geodesic_distances[goal_idx]
        self._agent_episode_distance = 0.0
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        goal_idx = episode.goal_idx
        success_distance = self._config.SUCCESS_DISTANCE + \
                           episode.goals[goal_idx].radius

        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[goal_idx].position
        )

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called
            and distance_to_target < success_distance
        ):
            if self._success_if_in_view:
                goal_stats = episode.goal_coord_in_camera
                _, _, _, xpx, ypx = goal_stats
                view_d = np.linalg.norm([xpx, ypx])
                if view_d < self._view_field:
                    ep_success = 1
            else:
                ep_success = 1

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_sensor
class GoalCoordInCamera(Sensor):
    r"""
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = 5
        self.fov = float(config.HFOV) * np.pi / 180.

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "goal_coord_in_camera"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        fov = self.fov

        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        _, phi = cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )

        theta = np.arcsin(
            direction_vector_agent[1]
            / np.linalg.norm(direction_vector_agent)
        )
        rho = np.linalg.norm(direction_vector_agent)

        dx = phi
        dy = -1 * theta

        fov_min = -1 * fov / 2
        fov_max = 1 * fov / 2

        xpx = dx / fov
        ypx = dy / fov

        if fov_max < dx < fov_min:
            xpx = -1

        if fov_max < dy < fov_min:
            ypx = -1

        return np.array([rho, -phi, theta, xpx, ypx], dtype=np.float32)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        goal_idx = getattr(episode, "goal_idx", 0)

        camera_state = self._sim.get_agent_state().sensor_states["rgb"]

        camera_position = camera_state.position
        rotation_world_camera = camera_state.rotation

        goal_position = np.array(episode.t_coord[episode.target_idx[goal_idx]],
                                 dtype=np.float32)

        stats = self._compute_pointgoal(
            camera_position, rotation_world_camera, goal_position
        )

        # TODO bad, but necessary for other sensors of measurements to use :Ds
        episode.goal_coord_in_camera = stats

        return stats


@registry.register_sensor
class GoalBBoxInCamera(GoalCoordInCamera):
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        super().__init__(sim=sim, config=config)
        self._dimensionality = 7

    def _compute_pointgoal_and_scale(
        self, source_position, source_rotation, goal_position, goal_scale
    ):
        prep = self._compute_pointgoal

        center = prep(source_position, source_rotation, goal_position)

        gx, gy, gz = goal_position - (abs(goal_scale) / 2)
        sx, sy, sz = abs(goal_scale)
        p1 = np.array([gx,      gy,      gz])
        p2 = np.array([gx + sx, gy,      gz])
        p4 = np.array([gx + sx, gy,      gz + sz])
        p6 = np.array([gx,      gy,      gz + sz])

        p3 = np.array([gx,      gy + sy, gz])
        p5 = np.array([gx + sx, gy + sy, gz])
        p8 = np.array([gx + sx, gy + sy, gz + sz])
        p7 = np.array([gx,      gy + sy, gz + sz])

        results = [
            prep(source_position, source_rotation, x) for x in [p1, p2, p4,
                                                                p6, p7, p8,
                                                                p3, p5]
        ]

        results = np.array(results)
        bbox_min = results.min(0)
        bbox_max = results.max(0)

        size = (bbox_max - bbox_min)[-2:]

        return np.concatenate([center, size])

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        goal_idx = getattr(episode, "goal_idx", 0)

        camera_state = self._sim.get_agent_state().sensor_states["rgb"]

        camera_position = camera_state.position
        rotation_world_camera = camera_state.rotation

        goal_position = np.array(episode.t_coord[episode.target_idx[goal_idx]],
                                 dtype=np.float32)

        goal_scale = np.array(episode.t_size[episode.target_idx[goal_idx]])

        stats = self._compute_pointgoal_and_scale(
            camera_position, rotation_world_camera, goal_position, goal_scale
        )
        crt_ep_stats = stats

        all_stats = []

        for idx, t_coord in enumerate(episode.t_coord):
            goal_position = np.array(t_coord, dtype=np.float32)

            goal_scale = np.array(episode.t_size[idx])

            stats = self._compute_pointgoal_and_scale(
                camera_position, rotation_world_camera, goal_position, goal_scale
            )
            all_stats.append(stats)

        all_stats = np.concatenate(all_stats)
        episode.all_stats = all_stats

        return crt_ep_stats

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "goal_bbox_in_camera"


@registry.register_sensor(name="ObjectGoalWithGPSCompassSensor")
class IntegratedObjectGoalGPSAndCompassSensor(PointGoalSensor):
    r"""
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "objectgoal_with_gps_compass"

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        goal_idx = getattr(episode, "goal_idx", 0)

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        goal_position = np.array(episode.t_coord[episode.target_idx[goal_idx]],
                                 dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
