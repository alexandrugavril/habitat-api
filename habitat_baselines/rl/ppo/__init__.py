#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.rl.ppo.policy import Net, PointNavBaselinePolicy, Policy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.models.simple_cnn_with_resnet import SimpleCNNResnet
from habitat_baselines.rl.models.simple_cnn_relu import SimpleCNNRelu

__all__ = ["PPO", "Policy", "Net", "PointNavBaselinePolicy"]

VISUAL_ENCODER_MODELS = dict({
    "SimpleCNN": SimpleCNN,
    "SimpleCNNRelu": SimpleCNNRelu,
    "SimpleCNNResnet": SimpleCNNResnet,
})
