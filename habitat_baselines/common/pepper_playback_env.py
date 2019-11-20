
from typing import Any, Dict, Optional, Type, List

import habitat
import roslibpy
from argparse import Namespace
from roslibpy import Message, Ros, Topic
from gym import Space, spaces
import quaternion
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

import numpy as np
import base64
import cv2
import pickle
from habitat import Config, Env
from habitat.core.dataset import Dataset, Episode

from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="PepperPlaybackEnv")
class PepperPlaybackEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        print("Initializing ENV")

        # Initialize ROS Bridge
        self._pepper_config = config.PEPPER

        self._ep_data = self.load_episode(self._pepper_config.EpisodePath)
        self._c_step = 0

        sim_config = config.TASK_CONFIG.SIMULATOR

        self._image_width = sim_config.RGB_SENSOR.WIDTH
        self._image_height = sim_config.RGB_SENSOR.HEIGHT
        self._rgb_batch_size = sim_config.RGB_SENSOR.BATCH

        self._norm_depth = sim_config.DEPTH_SENSOR.NORMALIZE_DEPTH
        self._depth_batch_size = sim_config.RGB_SENSOR.BATCH


        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        self._goal_sensor_dim = config.TASK_CONFIG.TASK.\
            POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

        self._collected_positions = set()

        self.init_obs_space()
        self.init_action_space()

        #super().__init__(self._core_env_config, dataset)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)

    def load_episode(self, path):
        data = pickle.load(open(path, "rb"))
        rotations = np.array([quaternion.from_euler_angles(
            np.roll(c_data['rotation'], 1))
                      for c_data in data])
        positions = np.array([c_data['position'] for c_data in data])

        prev_pos = positions[0]
        prev_rot = rotations[0]
        prev_heading = np.array([self._quat_to_xy_heading(
            prev_rot
        )])


        rel_positions = []
        for c_data in data:
            agent_position = c_data['position']
            agent_rotation = quaternion.from_euler_angles(np.roll(c_data['rotation'], 1))

            relative_pos = quaternion_rotate_vector(
                prev_rot.inverse(), agent_position - prev_pos
            )

            heading = np.array([self._quat_to_xy_heading(
                agent_rotation
            )])
            relative_heading = heading - prev_heading

            if np.abs(relative_heading) > np.pi:
                relative_heading = np.mod(relative_heading, 2 * np.pi *
                                          -np.sign(relative_heading))

            pos = np.array(
                [relative_pos[0], relative_pos[2]], dtype=np.float32
            )

            prev_pos = agent_position
            prev_heading = heading
            prev_rot = agent_rotation

            c_data['rel_position'] = np.concatenate([pos, relative_heading])

        return data

    def init_action_space(self):
        self.action_space = spaces.Discrete(3)

    def init_obs_space(self):
        self.observation_space = Namespace()
        rgb_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._image_height, self._image_width, 3 * self._rgb_batch_size),
            dtype=np.uint8,
        )
        if self._norm_depth:
            min_depth = 0
            max_depth = 1
        else:
            min_depth = 0
            max_depth = 255

        depth_space = spaces.Box(
            low=min_depth,
            high=max_depth,
            shape=(self._image_height, self._image_width, 1 * self._depth_batch_size),
            dtype=np.float,
        )

        pointgoal_with_gps_compass_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._goal_sensor_dim,),
            dtype=np.float32,
        )


        gps_compass_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._goal_sensor_dim,),
            dtype=np.float32,
        )

        self.observation_space.spaces = {
            "rgb": rgb_space,
            "depth": depth_space,
            self.goal_sensor_uuid: pointgoal_with_gps_compass_space,
            "gps_compass": gps_compass_space
        }

    @property
    def habitat_env(self) -> Env:
        return None

    @property
    def episodes(self) -> List[Type[Episode]]:
        return []

    @property
    def current_episode(self) -> Type[Episode]:
        ep = Namespace()
        ep.episode_id = 0
        ep.scene_id = 0
        return ep

    def seed(self, seed: Optional[int] = None) -> None:
        pass

    def render(self, mode: str = "rgb") -> np.ndarray:
        pass

    def close(self) -> None:
        pass

    def _wait_move_done(self):
        import time
        time.sleep(2)

    def resize_rgb(self, rgb):
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (self._image_width, self._image_height))
        return rgb

    def resize_depth(self, depth):
        depth = (depth.astype(np.float) / depth.max()).astype(np.float)
        depth = cv2.resize(depth, (self._image_width, self._image_height))
        depth = depth.reshape((self._image_height, self._image_width, 1))
        return depth

    def get_obs(self):
        assert self._c_step + self._rgb_batch_size < len(self._ep_data)

        data_batch = self._ep_data[self._c_step: self._c_step + self._rgb_batch_size]

        rgb = [self.resize_rgb(c_data['rgb']) for c_data in data_batch]
        depth = [self.resize_depth(c_data['depth']) for c_data in data_batch]
        sonar = [c_data['sonar'] for c_data in data_batch]
        position = [c_data['position'] for c_data in data_batch]
        rotation = [c_data['rotation'] for c_data in data_batch]
        action = [c_data['action'] for c_data in data_batch]
        rel_pos = [c_data['rel_position'] for c_data in data_batch]

        cv2.imshow("RGB", rgb[0])
        cv2.imshow("Depth", depth[0])
        cv2.waitKey(1)
        return {
            "rgb": np.concatenate(rgb, axis=2),
            "depth": np.concatenate(depth, axis=2),
            "sonar": np.stack(sonar),
            "position": np.stack(position),
            "rotation": np.stack(rotation),
            "action": np.stack(action),
            'rel_position': np.stack(rel_pos)
        }

    def reset(self):
        self._previous_action = None

        observations = self.get_obs()
        self._collected_positions = set()
        self._depth_buffer = []
        self._rgb_buffer = []
        self._c_step = 0

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]

        done = self._ep_data is not None and \
               (self._c_step + self._rgb_batch_size == len(self._ep_data))

        if done:
            self._c_step = 0

        observations = self.get_obs()
        reward = self.get_reward(observations)
        info = self.get_info(observations)

        self._c_step += 1

        return observations, reward, done, info

    def get_reward_range(self):
        return (
            0,
            1,
        )

    def get_position(self):
        c_pos = self._ep_data[self._c_step]['position']
        c_rot = self._ep_data[self._c_step]['rotation']

        return c_pos, c_rot

    def get_reward(self, observations):
        reward = 0
        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = self._ep_data is not None and \
               (self._c_step + self._rgb_batch_size == len(self._ep_data))

        return done

    def get_info(self, observations):
        map = np.random.rand(32, 32, 3)
        info = {
            "spl": 0.0,
            "top_down_map": {
                "map": np.array([[0, 0, 0]]),
                "valid_map": map,
                "explored_map": map,
                "ful_fog_of_war_mask": map,
                "fog_of_war_mask": None,
                "agent_map_coord": np.array([0, 0, 0]),
                "agent_angle": 0
            }
        }
        return info
