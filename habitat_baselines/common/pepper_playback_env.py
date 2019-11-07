
from typing import Any, Dict, Optional, Type, List

import habitat
import roslibpy
from argparse import Namespace
from roslibpy import Message, Ros, Topic
from gym import Space, spaces

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
        # Initialize ROS Bridge
        self._pepper_config = config.PEPPER

        self._ep_data = self.load_episode(self._pepper_config.EpisodePath)
        self._c_step = 0

        sim_config = config.TASK_CONFIG.SIMULATOR

        self._image_width = sim_config.RGB_SENSOR.WIDTH
        self._image_height = sim_config.RGB_SENSOR.HEIGHT
        self._norm_depth = sim_config.DEPTH_SENSOR.NORMALIZE_DEPTH

        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        self._goal_sensor_dim = config.TASK_CONFIG.TASK.\
            POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

        self._collected_positions = set()

        self.init_obs_space()
        self.init_action_space()

        #super().__init__(self._core_env_config, dataset)


    def load_episode(self, path):
        return pickle.load(open(path, "rb"))

    def init_action_space(self):
        self.action_space = spaces.Discrete(3)

    def init_obs_space(self):
        self.observation_space = Namespace()
        rgb_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._image_height, self._image_width, 3),
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
            shape=(self._image_height, self._image_width, 1),
            dtype=np.float,
        )

        pointgoal_with_gps_compass_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._goal_sensor_dim,),
            dtype=np.float32,
        )

        self.observation_space.spaces = {
            "rgb": rgb_space,
            "depth": depth_space,
            self.goal_sensor_uuid: pointgoal_with_gps_compass_space
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

    def get_obs(self):
        assert self._c_step < len(self._ep_data)

        c_data = self._ep_data[self._c_step]
        rgb = c_data['rgb']
        depth = c_data['depth']

        return {
            "rgb": rgb,
            "depth": depth
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

        observations = self.get_obs()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
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
        (x, y, z), (x, y, z, w) = self.get_position()
        reward = 0
        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = self._ep_data is not None and \
               (self._c_step + 1 == len(self._ep_data))
        return done

    def get_info(self, observations):
        map = np.random.rand(32, 32, 3)
        info = {
            "top_down_map": {
                "map": np.array([[0, 0, 0]]),
                "valid_map": map,
                "explored_map": map,
                "ful_fog_of_war_mask": map,
                "fog_of_war_mask": None,
                "agent_map_coord": np.array([0, 0, 0]),
                "agent_angle": 0
            },
            "position": self.get_position(),
            "action": self._ep_data[self._c_step]['action']
        }
        return info
