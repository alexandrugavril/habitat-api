#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict, List
import cv2
import matplotlib.pyplot as plt

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
    batch_obs_augment_aux,
)
from habitat_baselines.rl.ppo.aux_ppo import AuxPPO
from habitat_baselines.rl.ppo.aimas_reachability_policy import ExploreNavBaselinePolicy
from habitat_baselines.rl.ppo.aimas_reachability_policy_aux import ExploreNavBaselinePolicyAux
from habitat_baselines.rl.ppo.aimas_reachability_policy_aux_recurrentin import ExploreNavBaselinePolicyAuxRecurrentin
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


from habitat_baselines.rl.ppo.reachability_policy import ReachabilityPolicy


np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))


ACTOR_CRITICS = dict({
    "ExploreNavBaselinePolicy": ExploreNavBaselinePolicy,
    "ExploreNavBaselinePolicyAux": ExploreNavBaselinePolicyAux,
    "ExploreNavBaselinePolicyAuxRecurrentin": ExploreNavBaselinePolicyAuxRecurrentin,
})


@baseline_registry.register_trainer(name="ppoAimasReachability")
class PPOTrainerReachabilityAimas(PPOTrainer):
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
        cfg = self.config

        # Get object index
        logger.add_filehandler(cfg.LOG_FILE)
        self.prev_pos = []
        # -- Reachability stuff
        # First pass add rollouts detector_features memory
        train_reachability = cfg.RL.REACHABILITY.train
        self.r_enabled = cfg.RL.REACHABILITY.enabled
        if self.r_enabled:
            self.r_policy = ReachabilityPolicy(
                cfg.RL.REACHABILITY, self.envs.num_envs,
                self.envs.observation_spaces[0], device=self.device,
                with_training=train_reachability,
                tb_dir=cfg.TENSORBOARD_DIR
            )
            self.r_policy.to(self.device)
        else:
            self.r_policy = None

        # Add only intrinsic reward
        self.only_intrinsic_reward = cfg.RL.REACHABILITY.only_intrinsic_reward

        # Train PPO after rtrain
        self.skip_train_ppo_without_rtrain = \
            cfg.RL.REACHABILITY.skip_train_ppo_without_rtrain

        # Map output of aux prediction from actor critic to next step observation
        self.map_aux_to_obs = cfg.RL.PPO.actor_critic.map_aux_to_obs

        self.actor_critic = ACTOR_CRITICS[cfg.RL.PPO.actor_critic.type](
            cfg=cfg.RL.PPO.actor_critic,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=cfg.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            with_target_encoding=cfg.TASK_CONFIG.TASK.WITH_TARGET_ENCODING,
            device=self.device,
            reachability_policy=self.r_policy,
            visual_encoder=ppo_cfg.visual_encoder,
            drop_prob=ppo_cfg.visual_encoder_dropout,
            channel_scale=ppo_cfg.channel_scale,
        )
        self.actor_critic.to(self.device)
        self.actor_critic.map_aux_to_obs = self.map_aux_to_obs

        self.agent = AuxPPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            action_loss_coef=ppo_cfg.action_loss_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def _add_intrinsic_reward(self, batch: dict, actions: torch.tensor,
                              rewards: torch.tensor, masks: torch.tensor):
        if self.r_enabled:
            intrinsic_r = self.r_policy.act(batch, actions, rewards, masks)
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
        prev_pos = self.prev_pos

        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                aux_out
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )
            # TODO HACK HACK
            # print("<:><:><:><:><:><:> YOU HAVE A HACKY HACK "* 100)
            # aux_out["rel_start_pos_reg"] +=

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        send_actions = [a[0].item() for a in actions]

        outputs = self.envs.step(send_actions)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        # Check if env modified actions
        if "action" in observations[0]:
            for i in range(len(observations)):
                actions[i] = int(observations[i].pop("action"))

        # #
        # isc = 6
        # rgb = observations[isc]["rgb"][:, :, -3:].cpu().numpy()
        # depth = observations[isc]["depth"][:,:, -1].unsqueeze(2).cpu().numpy()
        # depth2 = observations[isc]["depth2"].cpu().numpy()
        #
        # # loc
        # loc = observations[isc]["gps_compass_start"]
        # print("New loc:", loc, actions[isc].item())
        # print("Sonar", depth2.min())
        # prev_pos.append(loc)
        # all_pos = np.array(prev_pos)
        #
        # img = cv2.resize(rgb, (0, 0), fx=2., fy=2)
        # # img = img.astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # depth = cv2.resize(depth, (0, 0), fx=2., fy=2)
        # depth2 = cv2.resize(depth2, (0, 0), fx=2., fy=2)
        # depth2 = depth2/2.5
        # cv2.imshow("OBS", img)
        # # cv2.imwrite(f"img_{int(time.time())}.jpg", img)
        # cv2.imshow("Depth", depth)
        # cv2.imshow("Depth2", depth2)
        #
        # cv2.waitKey(1)
        # #
        # plt.scatter(all_pos[:, 0], all_pos[:, 1])
        # plt.show()
        #
        env_time += time.time() - t_step_env

        t_update_stats = time.time()

        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        map_values = self._get_mapping(observations, aux_out)
        batch = batch_obs_augment_aux(observations, map_values=map_values, masks=masks)

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

    def _get_mapping(self, observations, aux_out):
        if len(self.map_aux_to_obs) > 0:
            map_values = dict()

            for aux_name, obs_name in self.map_aux_to_obs:
                map_values[obs_name] = aux_out[aux_name]

            return map_values
        return None

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        if not self.skip_train_ppo_without_rtrain or self.r_policy.is_trained:
            value_loss, action_loss, dist_entropy, aux_loss = \
                self.agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy, aux_loss = (0, 0, 0, 0)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            aux_loss
        )

    def add_new_based_on_cfg(self):
        # Add proximity sensor if COLLISION_REWARD_ENABLED &
        # COLLISION_DISTANCE > 0
        if self.config.RL.COLLISION_REWARD_ENABLED and \
            self.config.RL.COLLISION_DISTANCE > 0:
            self.config.defrost()
            if "PROXIMITY_SENSOR" not in self.config.TASK_CONFIG.TASK.SENSORS:
                self.config.TASK_CONFIG.TASK.SENSORS.append("PROXIMITY_SENSOR")
                self.config.TASK_CONFIG.TASK.PROXIMITY_SENSOR\
                    .MAX_DETECTION_RADIUS = \
                    self.config.RL.COLLISION_DISTANCE + 0.5
            self.config.freeze()

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.add_new_based_on_cfg()

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
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs_augment_aux(observations)

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

        train_steps = min(self.config.NUM_UPDATES,
                          self.config.HARD_NUM_UPDATES)
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(train_steps):
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

                delta_pth_time, value_loss, action_loss, dist_entropy,\
                    aux_loss = self._update_agent(ppo_cfg, rollouts)

                # TODO check if LR is init
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_go_reward.append(episode_go_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                value_names = ["value", "policy", "entropy"] + list(
                    aux_loss.keys())
                losses = [value_loss, action_loss, dist_entropy] + list(
                    aux_loss.values())

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
                    {k: l for l, k in zip(losses, value_names)},
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
                        logger.info(
                            f"Aux losses: {list(zip(value_names, losses))}"
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
        self.add_new_based_on_cfg()

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        # # Mostly for visualization
        # config.defrost()
        # config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
        # config.freeze()

        split = config.TASK_CONFIG.DATASET.SPLIT

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
        self.r_policy = self.agent.actor_critic.reachability_policy

        aux_models = self.actor_critic.net.aux_models

        other_losses = dict({k: torch.zeros(self.envs.num_envs, 1,
                                            device=self.device)
                             for k in aux_models.keys()})
        other_losses_action = dict({k: torch.zeros(self.envs.num_envs,
                                                   self.envs.action_spaces[0].n,
                                            device=self.device)
                                    for k in aux_models.keys()})

        # Config aux models for eval per item in batch
        for k, maux in aux_models.items():
            maux.set_per_element_loss()

        total_loss = 0

        if config.EVAL_MODE:
            self.agent.eval()
            self.r_policy.eval()

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
        batch = batch_obs_augment_aux(observations, self.device)

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
        stats_episodes_scenes = dict()  # dict of number of collected stats from
        # each scene
        max_test_ep_count = self.config.TEST_EPISODE_COUNT

        # TODO this should depend on number of scenes :(
        # TODO But than envs shouldn't be paused but fast-fwd to next scene
        # TODO We consider num envs == num scenes
        max_ep_per_env = max_test_ep_count / float(self.envs.num_envs)

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        video_log_int = self.config.VIDEO_OPTION_INTERVAL
        num_frames = 0

        # ---
        plot_pos = -1
        prev_true_pos = []
        prev_pred_pos = []

        while (
            len(stats_episodes) <= self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                prev_hidden = test_recurrent_hidden_states
                _, actions, _, test_recurrent_hidden_states, aux_out \
                    = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False
                )

                prev_actions.copy_(actions)

                if 'action' in batch:
                    prev_actions = batch['action'].unsqueeze(1).to(
                        actions.device).long()

                for k, v in aux_out.items():
                    loss = aux_models[k].calc_loss(
                        v,
                        batch,
                        prev_hidden,
                        prev_actions,
                        not_done_masks,
                        actions
                    )
                    total_loss += loss

                    if other_losses[k] is None:
                        other_losses[k] = loss
                    else:
                        other_losses[k] += loss
                    if len(prev_actions) == 1:
                        other_losses_action[k][0, prev_actions.item()] += \
                            loss.item()


                if plot_pos >= 0:
                    prev_true_pos.append(batch["gps_compass_start"][
                                             plot_pos].data[:2].cpu().numpy())
                    prev_pred_pos.append(aux_out["rel_start_pos_reg"][
                                             plot_pos].data.cpu().numpy() * 15)
                    if num_frames % 10 == 0:
                        xx, yy = [], []
                        for x, y in prev_true_pos:
                            xx.append(x)
                            yy.append(y)
                        plt.scatter(xx, yy, label="true_pos")
                        xx, yy = [], []
                        for x, y in prev_pred_pos:
                            xx.append(x)
                            yy.append(y)
                        plt.scatter(xx, yy, label="pred_pos")
                        plt.legend()
                        plt.show()
                        plt.waitforbuttonpress()
                        plt.close()

            num_frames += 1
            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            map_values = self._get_mapping(observations, aux_out)
            batch = batch_obs_augment_aux(observations, device=self.device, map_values=map_values,
                                          masks=not_done_masks)

            valid_map_size = [float(ifs["top_down_map"]["valid_map"].sum()) for ifs in infos]
            discovered_factor = [infos[ix]["top_down_map"]["explored_map"].sum() /
                                 valid_map_size[ix] for ix in range(len(
                infos))]

            seen_factor = [infos[ix]["top_down_map"]["ful_fog_of_war_mask"].sum() /
                           valid_map_size[ix] for ix in range(len(infos))]


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
                scene = next_episodes[i].scene_id

                if scene not in stats_episodes_scenes:
                    stats_episodes_scenes[scene] = 0

                if stats_episodes_scenes[scene] >= max_ep_per_env:
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
                    episode_stats["map_discovered"] = discovered_factor[i]
                    episode_stats["map_seen"] = seen_factor[i]

                    for k, v in other_losses.items():
                        episode_stats[k] = v[i].item() / num_frames
                        print("Loss:", k, episode_stats[k])
                        print("Action:", k, other_losses_action[k] / num_frames)

                        other_losses_action[k][i].fill_(0)
                        other_losses[k][i] = 0
                    num_frames = 0
                    print("-" * 100)

                    current_episode_reward[i] = 0
                    current_episode_go_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    stats_episodes_scenes[current_episodes[i].scene_id] += 1

                    if len(self.config.VIDEO_OPTION) > 0 and \
                        checkpoint_index % video_log_int == 0:
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
                    for k, v in observations[i].items():
                        if isinstance(v, torch.Tensor):
                            observations[i][k] = v.cpu().numpy()
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            # Pop done envs:
            if len(envs_to_pause) > 0:
                s_index = list(range(self.envs.num_envs))
                for idx in reversed(envs_to_pause):
                    s_index.pop(idx)

                current_episode_go_reward = current_episode_go_reward[s_index]
                for k, v in other_losses.items():
                    other_losses[k] = other_losses[k][s_index]


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

        episodes_agg_stats = dict()
        for k, v in aggregated_stats.items():
            episodes_agg_stats[k] = v / num_episodes
            logger.info(f"Average episode {k}: {episodes_agg_stats[k]:.6f}")

        for k, v in episodes_agg_stats.items():
            writer.add_scalars(
                f"eval_{k}",
                {f"{split}_average {k}": v},
                checkpoint_index
            )
            print(f"[{checkpoint_index}] average {k}", v)

        self.envs.close()
