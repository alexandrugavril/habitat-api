#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    generate_video,
    linear_decay,
)

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.utils import (
    batch_obs,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.aimas_reachability_policy import ExploreNavBaselinePolicy
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


from habitat_baselines.rl.ppo.reachability_policy import ReachabilityPolicy


@baseline_registry.register_trainer(name="ppoAimas")
class PPOTrainerAimas(PPOTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    def _setup_actor_critic_agent(self, ppo_cfg: Config, train: bool=True) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """

        # Get object index
        logger.add_filehandler(self.config.LOG_FILE)

        # -- Reachability stuff
        # First pass add rollouts detector_features memory

        self.reachability_policy = ReachabilityPolicy(self.config.RL.REACHABILITY,
                                                      self.envs.num_envs,
                                                      self.envs.observation_spaces[0],
                                                      device=self.device, with_training=train)
        self.reachability_policy.to(self.device)

        # Add only intrinsic reward
        self.only_intrinsic_reward = self.config.RL.REACHABILITY.only_intrinsic_reward
        # --

        self.actor_critic = ExploreNavBaselinePolicy(
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

    def _add_intrinsic_reward(self, batch: dict, actions: torch.tensor, rewards: torch.tensor,
                              masks: torch.tensor):

        intrinsic_r = self.reachability_policy.act(batch, actions, rewards, masks)
        rewards.add_(intrinsic_r)

        return rewards

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_go_reward,
        episode_rewards, episode_go_rewards, episode_counts
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

        detections = step_observation["r_features"]

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

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        current_episode_go_reward += rewards
        episode_go_rewards += (1 - masks) * current_episode_go_reward
        current_episode_go_reward *= masks

        # -- Add intrinsic Reward
        if self.only_intrinsic_reward:
            rewards.zero_()

        rewards = self._add_intrinsic_reward(batch, actions, rewards, masks)

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

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg, train=True)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            if sensor in batch:
                rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_go_rewards = torch.zeros(self.envs.num_envs, 1)  # Grid oracle rewars
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_go_reward = torch.zeros(self.envs.num_envs, 1) # Grid oracle rewars
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_go_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_go_reward,
                        episode_rewards,
                        episode_go_rewards,
                        episode_counts,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_go_reward.append(episode_go_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward", "reward_go"],
                    [window_episode_counts, window_episode_reward, window_episode_go_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalar(
                    "reward_go", deltas["reward_go"] / deltas["count"], count_steps
                )

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_go_rewards = (
                        window_episode_go_reward[-1] - window_episode_go_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f} reward_go: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                                (window_go_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        # config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            config, get_env_class(self.config.ENV_NAME)
        )
        self._setup_actor_critic_agent(ppo_cfg, train=False)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        # get name of performance metric, e.g. "spl"
        metric_name = self.config.TASK_CONFIG.TASK.MEASUREMENTS[0]
        metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
        measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
        assert measure_type is not None, "invalid measurement type {}".format(
            metric_cfg.TYPE
        )
        self.metric_uuid = measure_type(
            sim=None, task=None, config=None
        )._get_uuid()

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_go_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)

            current_episode_go_reward += rewards

            # -- Add intrinsic Reward
            if self.only_intrinsic_reward:
                rewards.zero_()

            rewards = self._add_intrinsic_reward(batch, actions, rewards, not_done_masks)

            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats[self.metric_uuid] = infos[i][
                        self.metric_uuid
                    ]
                    episode_stats["success"] = int(
                        infos[i][self.metric_uuid] > 0
                    )
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["reward_go"] = current_episode_go_reward[i].item()
                    current_episode_reward[i] = 0
                    current_episode_go_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name=self.metric_uuid,
                            metric_value=infos[i][self.metric_uuid],
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

        (
            self.envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        ) = self._pause_envs(
            envs_to_pause,
            self.envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_go_reward_mean = aggregated_stats["reward_go"] / num_episodes
        episode_metric_mean = aggregated_stats[self.metric_uuid] / num_episodes
        episode_success_mean = aggregated_stats["success"] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(f"Average episode reward GO: {episode_go_reward_mean:.6f}")
        logger.info(f"Average episode success: {episode_success_mean:.6f}")
        logger.info(
            f"Average episode {self.metric_uuid}: {episode_metric_mean:.6f}"
        )

        writer.add_scalars(
            "eval_reward",
            {"average reward": episode_reward_mean, "average go reward": episode_go_reward_mean},
            checkpoint_index,
        )
        writer.add_scalars(
            f"eval_{self.metric_uuid}",
            {f"average {self.metric_uuid}": episode_metric_mean},
            checkpoint_index,
        )
        writer.add_scalars(
            "eval_success",
            {"average success": episode_success_mean},
            checkpoint_index,
        )
        print(f"[{checkpoint_index}] average reward", episode_reward_mean)
        print(f"[{checkpoint_index}] average reward GO", episode_go_reward_mean)
        print(f"[{checkpoint_index}] average {self.metric_uuid}", episode_metric_mean)
        print(f"[{checkpoint_index}] average success", episode_success_mean)
        self.envs.close()