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

from habitat_baselines.rl.ppo.aux_relative_position import RelativePositionPredictor
from habitat_baselines.rl.ppo.aux_sonar import SonarPredictor

AUX_CLASSES = dict({
    "rel_pos": RelativePositionPredictor,
    "sonar": SonarPredictor
})


class ExploreNavBaselinePolicyAux(Policy):
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
            ExploreNavBaselineNetAux(
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


class ExploreNavBaselineNetAux(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, cfg, observation_space, hidden_size, goal_sensor_uuid,
                 with_target_encoding, device, visual_encoder="SimpleCNN",
                 drop_prob=0.5, channel_scale=1):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.with_target_encoding = with_target_encoding
        num_recurrent_layers = getattr(cfg, "num_recurrent_layers", 1)
        rnn_type = getattr(cfg, "rnn_type", "GRU")

        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size

        self.visual_encoder = VISUAL_ENCODER_MODELS[visual_encoder](
            observation_space, hidden_size, drop_prob=drop_prob,
            channel_scale=channel_scale)

        visual_feat_size = 0 if self.is_blind else self._hidden_size
        rnn_out_size = self._hidden_size
        t_enc_size = self._n_input_goal if with_target_encoding else 0

        self.aux_models = aux_models = torch.nn.ModuleDict({})
        for aux_type in cfg.aux:
            aux_cfg = getattr(cfg, AUX_CLASSES[aux_type].__name__)
            aux_models[aux_type] = AUX_CLASSES[aux_type](
                aux_cfg, visual_feat_size, t_enc_size, rnn_out_size)

        print(rnn_type, num_recurrent_layers)
        self.state_encoder = RNNStateEncoder(
            visual_feat_size +
            t_enc_size,
            self._hidden_size,
            num_layers=num_recurrent_layers,
            rnn_type=rnn_type
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        target_encoding = None
        perception_embed = None

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

        return x, rnn_hidden_states, aux_out
