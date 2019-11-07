
from typing import Any, Dict, Optional, Type, List

import habitat
import roslibpy
from argparse import Namespace
from roslibpy import Message, Ros, Topic
from gym import Space, spaces

import numpy as np
import base64
import cv2
from habitat import Config, Env
from habitat.core.dataset import Dataset, Episode

from habitat_baselines.common.baseline_registry import baseline_registry


def _get_movement_ros_message(fw_step, r_step):
    frame_id = 'base_footprint'
    m = Message({
        'header': {
            'frame_id': frame_id
        },
        'pose': {
            'position': {
                'x': fw_step,
                'y': 0,
                'z': 0
            },
            'orientation': {
                'x': 0,
                'y': 0,
                'z': r_step,
                'w': 1
            }
        }
    })
    return m


@baseline_registry.register_env(name="PepperRLExploration")
class PepperRLExplorationEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        # Initialize ROS Bridge
        self._pepper_config = config.PEPPER
        self._ros = roslibpy.Ros(host='localhost', port=9090)
        self._ros.run()
        assert self._ros.is_connected, "ROS not connected"

        sim_config = config.TASK_CONFIG.SIMULATOR

        self._image_width = sim_config.RGB_SENSOR.WIDTH
        self._image_height = sim_config.RGB_SENSOR.HEIGHT
        self._norm_depth = sim_config.DEPTH_SENSOR.NORMALIZE_DEPTH

        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        self._goal_sensor_dim = config.TASK_CONFIG.TASK.\
            POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

        self._buffer_size = self._pepper_config.BufferSize
        self._forward_step = self._pepper_config.ForwardStep
        self._turn_step = self._pepper_config.TurnStep

        self._depth_buffer = []
        self._rgb_buffer = []
        self._collected_positions = set()

        # Subscribe to RGB and Depth topics
        # RGBTopic: '/pepper_robot/naoqi_driver/camera/front/image_raw'
        # DepthTopic: '/pepper_robot/naoqi_driver/camera/depth/image_raw'
        # MoveTopic: '/move_base_simple/goal'
        self._listener_rgb = Topic(self._ros,
                                   self._pepper_config.RGBTopic,
                                   'sensor_msgs/Image')
        self._listener_depth = Topic(self._ros,
                                     self._pepper_config.DepthTopic,
                                     'sensor_msgs/Image')
        self._publisher_move = Topic(self._ros,
                                     self._pepper_config.MoveTopic,
                                     'geometry_msgs/PoseStamped')

        self._listener_rgb.subscribe(lambda message:
                                     self._fetch_rgb(message))
        self._listener_depth.subscribe(lambda message:
                                       self._fetch_depth(message))

        self.init_obs_space()
        self.init_action_space()

        #super().__init__(self._core_env_config, dataset)

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
        self._ros.close()

    def _fetch_rgb(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint8)
        img = img.reshape((message['height'], message['width'], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self._image_width, self._image_height))
        self._rgb_buffer.append(img)

        if self._buffer_size != -1:
            if len(self._rgb_buffer) > self._buffer_size:
                self._rgb_buffer.pop(0)

    def _fetch_depth(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint16)
        img = img.reshape((message['height'], message['width'], 1))
        img = (img.astype(np.float) / 9460.0).astype(np.float)
        img = cv2.resize(img, (self._image_width, self._image_height))
        img = img.reshape((self._image_height, self._image_width, 1))
        self._depth_buffer.append(img)

        if self._buffer_size != -1:
            if len(self._depth_buffer) > self._buffer_size:
                self._depth_buffer.pop(0)

    def _wait_move_done(self):
        import time
        time.sleep(1.5)

    def _send_command(self, action):
        action = action['action']

        if action == 0:
            print("Action:", "Forward", self._forward_step, self._turn_step)
            m = _get_movement_ros_message(self._forward_step, 0)
            self._publisher_move.publish(m)
        elif action == 1:
            print("Action:", "Left", 0, self._turn_step)
            m = _get_movement_ros_message(0, self._turn_step)
            self._publisher_move.publish(m)
        elif action == 2:
            print("Action:", "Right", 0, self._turn_step)
            m = _get_movement_ros_message(0, -1 * self._turn_step)
            self._publisher_move.publish(m)
        self._wait_move_done()

    def get_obs(self):
        if len(self._rgb_buffer) > 0:
            rgb = self._rgb_buffer[-1]
        else:
            rgb = np.random.rand(self._image_height, self._image_width, 3)

        if len(self._depth_buffer) > 0:
            depth = self._depth_buffer[-1]
        else:
            depth = np.random.rand(self._image_height, self._image_width, 1)
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth)
        cv2.waitKey(1)
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

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        self._send_command(self._previous_action)

        observations = self.get_obs()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def get_reward_range(self):
        return (
            0,
            1,
        )

    def get_position(self):
        return 0, 0, 0

    def get_reward(self, observations):
        x, y, z = self.get_position()
        reward = 0
        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = False
        # if self._env.episode_over or self._episode_success():
        #     done = True
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
            }
        }
        return info
