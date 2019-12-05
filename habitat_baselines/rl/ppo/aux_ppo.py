#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

from habitat_baselines.rl.ppo import PPO
EPS_PPO = 1e-5


class AuxPPO(PPO):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        action_loss_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):
        super().__init__(actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_normalized_advantage=use_normalized_advantage)

        self.action_loss_coef = action_loss_coef

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)
        aux_models = self.actor_critic.net.aux_models

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        other_losses = dict({k: 0 for k in aux_models.keys()})

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_out
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss * self.action_loss_coef
                    - dist_entropy * self.entropy_coef
                )

                for k, v in aux_out.items():
                    loss = aux_models[k].calc_loss(
                        v,
                        obs_batch,
                        recurrent_hidden_states_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch
                    )
                    total_loss += loss
                    other_losses[k] += loss.item()

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                # Update observations
                # Tensors of size tensors to (num_steps * num_envs_per_batch, ...)

                # HACKY update
                if len(self.actor_critic.map_aux_to_obs) > 0:
                    aux_out_rel = aux_out["rel_start_pos_reg"].view(256, 3, 2)
                    obs_view = obs_batch["empty_sensor"].view(256, 3, 2)
                    obs_view[1:].copy_(aux_out_rel[:-1])

                    obs_batch["empty_sensor"].data *= masks_batch.data

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        for k, v in other_losses.items():
            other_losses[k] = v / num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
               other_losses
