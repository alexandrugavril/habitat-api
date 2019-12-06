from typing import Any, Dict, Optional, Type, Union
import numpy as np
import torch
import collections

import habitat
from habitat import Config, Dataset, SimulatorActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.augmentation import PhotometricDistort
from habitat_baselines.common.augmentation import RandomMove


class AugmentEnv(habitat.RLEnv):
    def __init__(self,  config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.cfg_transform = config.TRANSFORM
        self.cfg_rgb = config.TRANSFORM_RGB
        self.cfg_depth = config.TRANSFORM_DEPTH
        self.batch_input = config.BATCH_INPUT
        self._eval_mode = config.EVAL_MODE

        self.photo_distort = PhotometricDistort()
        min_scale = self.cfg_transform.min_scale
        max_scale = self.cfg_transform.max_scale
        self.move_image = RandomMove(min_scale, max_scale)

        self._multi_batch_rgb = collections.deque(maxlen=self.batch_input)
        self._multi_batch_depth = collections.deque(maxlen=self.batch_input)

        self._fixed_distribution = config.RL.PPO.actor_critic.fixed_distribution
        self._prev_action = None
        self._explore_heuristic = len(self._fixed_distribution) > 0
        self._prev_obs = None
        self._fwd_steps = 0
        self._select_rotate = np.random.choice([1, 2])

    def _process_obs(self, obs):

        if self._eval_mode:
            return obs

        if self.cfg_depth.ENABLED:
            obs["depth"] = self.process_depth(obs["depth"])

        if self.cfg_rgb.ENABLED:
            obs["rgb"] = self.process_rgb(obs["rgb"])

        if self.cfg_transform.ENABLED:
            data = torch.cat([obs["rgb"].float(), obs["depth"]], dim=2)
            data = self.move_image(data)
            obs["rgb"] = data[:, :, :3]
            obs["depth"] = data[:, :, -1].unsqueeze(2)

        obs["rgb"] = obs["rgb"].byte()

        if self.batch_input > 1:
            self._multi_batch_rgb.append(obs["rgb"])
            self._multi_batch_depth.append(obs["depth"])

            obs["rgb"] = torch.cat(list(self._multi_batch_rgb), axis=2)
            obs["depth"] = torch.cat(list(self._multi_batch_depth), axis=2)

        return obs

    def process_depth(self, depth):
        std_noise = self.cfg_depth.std_noise
        noise_interval = self.cfg_depth.noise_interval
        depth_min_th = self.cfg_depth.depth_min_th
        depth_max_th = self.cfg_depth.depth_max_th

        zone = np.random.rand()
        depth *= ((depth < (zone - noise_interval)) |
                  (depth > (zone + noise_interval)))

        depth_noise = torch.normal(1, std_noise, depth.shape,
                                   device=depth.device)
        depth *= depth_noise
        depth.clamp_(0., 1.)

        depth *= ((depth > depth_min_th) &
                  (depth < depth_max_th))

        return depth

    def process_rgb(self, rgb):
        rgb = self.photo_distort(rgb.float())
        rgb.clamp_(0, 255)

        return rgb

    def reset(self):
        self._prev_action = None
        self._select_rotate = np.random.choice([1, 2])

        observations = super().reset()

        if self.batch_input > 1:
            rgb = torch.zeros_like(observations["rgb"])
            self._multi_batch_rgb = collections.deque(
                [rgb] * self.batch_input, maxlen=self.batch_input)

            depth = torch.zeros_like(observations["depth"])
            self._multi_batch_depth = collections.deque(
                [depth] * self.batch_input, maxlen=self.batch_input)

        observations = self._process_obs(observations)
        self._prev_obs = observations

        return observations

    def step(self, *args, **kwargs):
        if self._explore_heuristic:
            sonar = self._prev_obs["depth2"]
            nonzero = sonar != 0

            if nonzero.any():
                sonar = sonar[sonar != 0].min()
            else:
                sonar = 2.8

            movement = self._prev_obs["gps_compass"][:2]
            movement = np.linalg.norm(movement)

            if sonar <= 0.3 or (self._prev_action == 0 and movement < 0.15):
                kwargs["action"]["action"] = self._select_rotate
            else:
                kwargs["action"]["action"] = np.random.choice(
                    [0, 1, 2], p=self._fixed_distribution)

            if kwargs["action"]["action"] == 0:
                self._fwd_steps += 1
                if self._fwd_steps % 3 == 0:
                    self._select_rotate = np.random.choice([1, 2])
            else:
                self._fwd_steps = 0

        observation, reward, done, info = super().step(*args, **kwargs)
        self._prev_obs = observation
        observation = self._process_obs(observation)

        self._prev_action = kwargs["action"]["action"]
        if self._explore_heuristic:
            observation["action"] = self._prev_action

        return observation, reward, done, info
