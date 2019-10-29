#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from gym import spaces
import numpy as np
import torch

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.utils import (
    batch_obs,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.aimas_policy import ObjectClassNavBaselinePolicy
from habitat_baselines.rl.ppo.aimas_policy import YoloDetector
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


@baseline_registry.register_trainer(name="ppoAimas")
class PPOTrainerAimas(PPOTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    def _setup_actor_critic_agent(self, ppo_cfg: Config, train: bool=True) \
        -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        # TODO Ugly for YOLO TO work
        # torch.backends.cudnn.enabled = True
        torch.cuda.set_device(self.device.index)

        # Get object index
        logger.add_filehandler(self.config.LOG_FILE)

        # First pass add rollouts detector_features memory
        self.envs.observation_spaces[0].spaces["detector_features"] = \
            spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(765 // (3 * 3), 32, 32),
            dtype=np.float32,
        )

        # generate feature convertor to Yolo class
        self.detector_class_select = YoloDetector.class_selector()

        self.actor_critic = ObjectClassNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            detector_config=self.config.DETECTOR,
            device=self.device
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, episode_rewards, episode_counts
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions

        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            # Collect detector features
            step_observation.pop("detector_features")

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        detections = step_observation["detector_features"]

        # Add to rollout memory to do inference with detector just once
        rollouts.observations["detector_features"][rollouts.step].copy_(
            detections)

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        # Add Reward from detection

        pred_conf = detections[:, 4]
        pred_cls = detections[:, 5:]

        detect_class = self.detector_class_select[
            batch["goalclass"].nonzero()[:, 1]]
        class_weight = pred_cls[detect_class]
        reward_class = class_weight.view(class_weight.size(0), -1).mean(dim=1)

        # rewards += reward_class.unsqueeze(1).cpu()
        #

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_counts += 1 - masks
        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs
