
from typing import Any, Dict, Optional, Type, Union

import habitat
import roslibpy
from roslibpy import Message, Ros, Topic

import numpy as np
import base64
import cv2
from habitat import Config, Dataset
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
                'z': 0,
                'w': r_step
            }
        }
    })
    return m


@baseline_registry.register_env(name="PepperRLExploration")
class PepperRLExplorationEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        # Initialize ROS Bridge
        self._pepper_config = config.Pepper
        self._ros = roslibpy.Ros(host='localhost', port=9090)
        self._ros.run()
        assert self._ros.is_connected, "ROS not connected"

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
                                       self.fetch_depth(message))

        self._core_env_config = config.TASK_CONFIG

        self._previous_action = None
        self._grid_resolution = self._rl_config.REACHABILITY.grid_resolution

        super().__init__(self._core_env_config, dataset)

    def _fetch_rgb(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint8)
        img = img.reshape((message['height'], message['width'], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._rgb_buffer.append(img)

        if self._buffer_size != -1:
            if len(self._rgb_buffer) > self._buffer_size:
                self._rgb_buffer.pop(0)

    def _fetch_depth(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint16)
        img = img.reshape((message['height'], message['width'], 1))
        self._depth_buffer.append(img)

        if self._buffer_size != -1:
            if len(self._depth_buffer) > self._buffer_size:
                self._depth_buffer.pop(0)

    def _wait_move_done(self):
        pass

    def _send_command(self, action):
        print("Action:", action)
        if action == "MOVE_FORWARD":
            m = _get_movement_ros_message(self._forward_step, 0)
            self._publisher_move.publish(m)
        elif action == 'TURN_LEFT':
            m = _get_movement_ros_message(0, self._turn_step)
            self._publisher_move.publish(m)
        elif action == 'TURN_RIGHT':
            m = _get_movement_ros_message(0, -1 * self._turn_step)
            self._publisher_move.publish(m)

        self._wait_move_done()

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

        observations = {
            'rgb': self._rgb_buffer[-1],
            'depth': self._depth_buffer[-1]
        }

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def get_reward_range(self):
        return (
            0,
            1,
        )

    def get_reward(self, observations):
        x, y, z = self.get_position()

        reward = 0

        quantized_x = int(x / self._grid_resolution)
        quantized_y = int(y / self._grid_resolution)
        quantized_z = int(z / self._grid_resolution)
        position_id = (quantized_x, quantized_y, quantized_z)

        if position_id not in self._collected_positions:
            reward += 1
            self._collected_positions.add(position_id)

        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = False
        # if self._env.episode_over or self._episode_success():
        #     done = True
        return done

    def get_info(self, observations):
        return {}
