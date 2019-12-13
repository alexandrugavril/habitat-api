
from typing import Any, Dict, Optional, Type, List

import time
import habitat
import roslibpy
from argparse import Namespace
from roslibpy import Message, Ros, Topic
from gym import Space, spaces

from habitat_baselines.config.default import get_config
import numpy as np
import base64
import cv2
from habitat import Config, Env
from habitat.core.dataset import Dataset, Episode

from habitat_baselines.common.baseline_registry import baseline_registry
import math


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


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

class PepperRecord:
    def __init__(self):
        # Initialize ROS Bridge
        config = get_config()
        self._pepper_config = config.PEPPER
        self._ros = roslibpy.Ros(host='localhost', port=9090)
        self._ros.run()
        assert self._ros.is_connected, "ROS not connected"

        sim_config = config.TASK_CONFIG.SIMULATOR

        self._image_width = sim_config.RGB_SENSOR.WIDTH
        self._image_height = sim_config.RGB_SENSOR.HEIGHT
        self._norm_depth = sim_config.DEPTH_SENSOR.NORMALIZE_DEPTH

        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        print(self.goal_sensor_uuid)
        self._goal_sensor_dim = config.TASK_CONFIG.TASK.\
            POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

        self._buffer_size = self._pepper_config.BufferSize
        self._forward_step = self._pepper_config.ForwardStep
        self._turn_step = self._pepper_config.TurnStep

        self._depth_buffer = []
        self._rgb_buffer = []

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

        self._listener_rgb.subscribe(lambda message:
                                     self._fetch_rgb(message))
        self._listener_depth.subscribe(lambda message:
                                       self._fetch_depth(message))

        self._fresh_rgb = False
        self._fresh_depth = False

        self.image_pairs = []
        self.last_rgb = time.time()
        self.last_depth = time.time()

    def close(self) -> None:
        self._ros.close()

    def rgb_message_to_cv(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint8)
        img = img.reshape((message['height'], message['width'], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self._image_width, self._image_height))
        return img

    def depth_message_to_cv(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint16)
        img = img.reshape((message['height'], message['width'], 1))
        img = (img.astype(np.float) / 9100).astype(np.float)
        img = img.clip(0.0, 1.0)
        img = cv2.resize(img, (self._image_width, self._image_height))
        img = img.reshape((self._image_height, self._image_width, 1))
        return img

    def _fetch_rgb(self, message):
        self._rgb_buffer.append(message)
        print("RGB:", time.time() - self.last_rgb)
        self.last_rgb = time.time()

    def _fetch_depth(self, msg):
        self._depth_buffer.append(msg)
        print("DEPTH:", time.time() - self.last_depth)
        self.last_depth = time.time()

        if len(self._depth_buffer) > 1:
            message = self._depth_buffer[-2]

            t_stamp_depth = message['header']['stamp']['secs']
            rgb_in_same_second = [rgb for rgb in self._rgb_buffer
                                  if (rgb['header']['stamp']['secs'] -
                                      t_stamp_depth) == 0]

            if len(rgb_in_same_second) > 0:
                diff_in_nsecs = abs(message['header']['stamp']['nsecs'] -
                      rgb_in_same_second[-1]['header']['stamp']['nsecs'])

                if diff_in_nsecs > 200000000:
                    return

                depth_img = self.depth_message_to_cv(message)
                rgb_img = self.rgb_message_to_cv(rgb_in_same_second[0])

                self.image_pairs.append({'rgb': rgb_img, 'depth': depth_img})
                print(len(self.image_pairs))


pepper = PepperRecord()
while len(pepper.image_pairs) < 10:
    pass

for pair in pepper.image_pairs:
    cv2.imshow("RGB", pair['rgb'])
    cv2.imshow("Depth", pair['depth'])
    cv2.waitKey(0)




