
from typing import Any, Dict, Optional, Type, List

import time
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
        print(self.goal_sensor_uuid)
        self._goal_sensor_dim = config.TASK_CONFIG.TASK.\
            POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

        self._buffer_size = self._pepper_config.BufferSize
        self._forward_step = self._pepper_config.ForwardStep
        self._turn_step = self._pepper_config.TurnStep

        self._sonar_buffer = []
        self._depth_buffer = []
        self._rgb_buffer = []
        self._pose_buffer = []
        self._goal_buffer = []
        self._odom_buffer = []
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
        self._listener_pose = Topic(self._ros,
                                    self._pepper_config.PoseTopic,
                                    'geometry_msgs/PoseStamped')
        self._listener_odom = Topic(self._ros,
                                    self._pepper_config.OdomTopic,
                                    'nav_msgs/Odometry')
        self._listener_goal = Topic(self._ros,
                                    self._pepper_config.GoalTopic,
                                    'geometry_msgs/PoseStamped')
        self._listener_sonar = Topic(self._ros,
                                     self._pepper_config.SonarTopic,
                                     'sensor_msgs/Range')

        self._listener_rgb.subscribe(lambda message:
                                     self._fetch_rgb(message))
        self._listener_depth.subscribe(lambda message:
                                       self._fetch_depth(message))
        self._listener_pose.subscribe(lambda message:
                                      self._fetch_pose(message))
        self._listener_odom.subscribe(lambda message:
                                      self._fetch_odom(message))
        self._listener_goal.subscribe(lambda message:
                                      self._fetch_goal(message))
        self._listener_sonar.subscribe(lambda message:
                                       self._fetch_sonar(message))

        self._fresh_sonar = False
        self._fresh_pose = False
        self._fresh_rgb = False
        self._fresh_depth = False
        self._fresh_odom = False

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

    def _fetch_goal(self, message):
        self._goal_buffer.append(message)

    def _fetch_odom(self, message):
        self._odom_buffer.append(message)
        self._fresh_odom = True

        if self._buffer_size != -1:
            if len(self._odom_buffer) > self._buffer_size:
                self._odom_buffer.pop(0)

    def _fetch_pose(self, message):
        import time
        self._pose_buffer.append(message)
        self._fresh_pose = True

        if self._buffer_size != -1:
            if len(self._pose_buffer) > self._buffer_size:
                self._pose_buffer.pop(0)

    def _fetch_sonar(self, message):
        self._sonar_buffer.append(message)
        self._fresh_sonar = True

        if self._buffer_size != -1:
            if len(self._sonar_buffer) > self._buffer_size:
                self._sonar_buffer.pop(0)

    def _fetch_rgb(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint8)
        img = img.reshape((message['height'], message['width'], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self._image_width, self._image_height))
        self._rgb_buffer.append(img)
        self._fresh_rgb = True


        if self._buffer_size != -1:
            if len(self._rgb_buffer) > self._buffer_size:
                self._rgb_buffer.pop(0)

    def _fetch_depth(self, message):
        img = np.frombuffer(base64.b64decode(message['data']), np.uint16)
        img = img.reshape((message['height'], message['width'], 1))
        img = (img.astype(np.float) / 9100).astype(np.float)
        img = img.clip(0.0, 1.0)
        img = cv2.resize(img, (self._image_width, self._image_height))
        img = img.reshape((self._image_height, self._image_width, 1))
        self._depth_buffer.append(img)
        self._fresh_depth = True

        if self._buffer_size != -1:
            if len(self._depth_buffer) > self._buffer_size:
                self._depth_buffer.pop(0)

    def _wait_move_done(self):
        import time
        time.sleep(4)

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

    def wait_all_fresh(self):
        print(self._fresh_depth, self._fresh_rgb, self._fresh_pose,
              self._fresh_sonar, self._fresh_odom)
        return self._fresh_depth and self._fresh_rgb and self._fresh_pose \
               and self._fresh_sonar and self._fresh_odom

    def get_obs(self):
        while not self.wait_all_fresh():
            print("Did not receive new obs.")
            time.sleep(0.5)

        rgb = self._rgb_buffer[-1]
        self._fresh_rgb = False

        depth = self._depth_buffer[-1]
        self._fresh_depth = False

        if self._pepper_config.DisplayImages:
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth)
            cv2.waitKey(1)

        robot_position, robot_rotation = self.get_position()
        goal_position = self.get_goal()

        dist = np.linalg.norm(robot_position - goal_position)
        inc_y = goal_position[1] - robot_position[1]
        inc_x = goal_position[0] - robot_position[0]
        angle_between_robot_and_goal = math.atan2(inc_y, inc_x)

        angle = robot_rotation[0] - angle_between_robot_and_goal

        angle = -1 * angle

        print(dist, angle)

        return {
            "rgb": rgb,
            "depth": depth,
            "robot_position": robot_position,
            "robot_rotation": robot_rotation,
            "sonar": self.get_sonar(),
            "odom": self.get_odom(),
            "gps_with_pointgoal_compass": [dist, angle]
        }

    def reset(self):
        self._previous_action = None

        self._collected_positions = set()
        self._depth_buffer = []
        self._rgb_buffer = []
        observations = self.get_obs()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

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

    def get_goal(self):
        if len(self._goal_buffer) == 0:
            return np.array([0, 0, 0])
        else:
            position = self._goal_buffer[-1]['pose']['position']
            c_pos = np.array([position['x'], position['y'], position['z']])
            return c_pos

    def get_sonar(self):
        self._fresh_sonar = False
        sonar = self._sonar_buffer[-1]['range']
        return sonar


    def get_odom(self):
        print(self._odom_buffer[-1])
        position = self._odom_buffer[-1]['pose']['pose']['position']
        rotation = self._odom_buffer[-1]['pose']['pose']['orientation']

        c_pos = np.array([position['x'], position['y'], position['z']])
        c_rot = np.array(quaternion_to_euler(rotation['x'], rotation['y'],
                                             rotation['z'], rotation['w']))
        self._fresh_pose = False

        return c_pos, c_rot

    def get_position(self):

        position = self._pose_buffer[-1]['pose']['position']
        rotation = self._pose_buffer[-1]['pose']['orientation']

        c_pos = np.array([position['x'], position['y'], position['z']])
        c_rot = np.array(quaternion_to_euler(rotation['x'], rotation['y'],
                 rotation['z'], rotation['w']))
        self._fresh_pose = False

        return c_pos, c_rot

    def get_reward(self, observations):
        #(x, y, z), (x, y, z, w) = self.get_position()
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
