from typing import Any, Dict, Optional, Type, Union
import numpy as np
import torch

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
        self._eval_mode = config.EVAL_MODE

        self.photo_distort = PhotometricDistort()
        min_scale = self.cfg_transform.min_scale
        max_scale = self.cfg_transform.max_scale
        self.move_image = RandomMove(min_scale, max_scale)

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

        observations = super().reset()
        observations = self._process_obs(observations)

        return observations

    def step(self, *args, **kwargs):

        observation, reward, done, info = super().step(*args, **kwargs)
        observation = self._process_obs(observation)

        return observation, reward, done, info
