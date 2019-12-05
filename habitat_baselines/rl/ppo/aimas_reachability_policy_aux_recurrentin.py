#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch

from habitat_baselines.common.utils import FixedDistributionNet
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net
from habitat_baselines.rl.ppo import VISUAL_ENCODER_MODELS
from habitat_baselines.rl.ppo import AUX_CLASSES
from habitat_baselines.rl.ppo.aimas_reachability_policy_aux import \
    ExploreNavBaselinePolicyAux, ExploreNavBaselineNetAux


class ExploreNavBaselinePolicyAuxRecurrentin(Policy):
    def __init__(
        self,
        cfg,
        observation_space,
        action_space,
        goal_sensor_uuid,
        with_target_encoding,
        device,
        hidden_size=512,
        reachability_policy=None,
        visual_encoder=None,
        drop_prob=0.5,
        channel_scale=1,
    ):
        super().__init__(
            ExploreNavBaselineNetAuxRecurrentin(
                cfg=cfg,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                with_target_encoding=with_target_encoding,
                device=device,
                visual_encoder=visual_encoder,
                drop_prob=drop_prob,
                channel_scale=channel_scale,
            ),
            action_space.n,
        )

        self.reachability_policy = reachability_policy

        if cfg.fixed_distribution:
            assert action_space.n == len(cfg.fixed_distribution), "Action " \
                                                                  "space " \
                                                                  "should " \
                                                                  "have the " \
                                                                  "same dim"
            self.action_distribution = \
                FixedDistributionNet(cfg.fixed_distribution)


class ExploreNavBaselineNetAuxRecurrentin(ExploreNavBaselineNetAux):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = None
        perception_embed = None

        if prev_actions.size(0) == rnn_hidden_states.size(1):
            x = []

            if self.with_target_encoding:
                target_encoding = self.get_target_encoding(observations)
                target_encoding = target_encoding  # /30. # TODO MAYBE NEED TO NORM
                x = [target_encoding]

            if not self.is_blind:
                perception_embed = self.visual_encoder(observations)
                x = [perception_embed] + x

            x = torch.cat(x, dim=1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

            aux_out = dict({})
            for aux, aux_model in self.aux_models.items():
                aux_out[aux] = aux_model(
                    observations, prev_actions, masks,
                    perception_embed, target_encoding, x
                )
        else:

            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = rnn_hidden_states.size(1)
            t = int(prev_actions.size(0) / n)
            ret_x = []
            aux_out = dict({})

            masks = masks.view(t, n, 1)

            if not self.is_blind:
                perception_embed = self.visual_encoder(observations)
                perception_embed = perception_embed.view(t, n, -1)

            if self.with_target_encoding:
                target_encoding = self.get_target_encoding(observations)
                target_encoding = target_encoding  # /30. # TODO MAYBE NEED TO NORM
                target_encoding = target_encoding.view(t, n, -1)

            target_encoding = target_encoding[0]
            for step in range(t):
                x = []
                mask = masks[step]

                if self.with_target_encoding:
                    x = [target_encoding]

                if not self.is_blind:
                    x = [perception_embed[step]] + x

                x = torch.cat(x, dim=1)
                x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, mask)
                ret_x.append(x)

                for aux, aux_model in self.aux_models.items():
                    if aux not in aux_out:
                        aux_out[aux] = []
                    aux_out[aux].append(aux_model(
                        observations, prev_actions, masks,
                        perception_embed, target_encoding, x
                    ))
                target_encoding = aux_out["rel_start_pos_reg"][-1]

            x = torch.cat(ret_x, dim=0)
            for aux, v in aux_out.items():
                aux_out[aux] = torch.cat(v, dim=0)

        return x, rnn_hidden_states, aux_out
