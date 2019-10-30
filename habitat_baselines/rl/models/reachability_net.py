r"""
    Credits and part of script from:
    https://github.com/facebookresearch/habitat-api
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ReachabilityFeatures(nn.Module):
    def __init__(self, observation_space, pretrained=True):
        super().__init__()
        assert "rgb" in observation_space.spaces

        self._net = net = resnet18(pretrained=pretrained)

        self.use_depth = False
        if "depth" in observation_space.spaces:
            _n_depth = observation_space.spaces["depth"].shape[2]
            self.use_depth = True
            self.depth_conv = nn.Conv2d(_n_depth, net.conv1.out_channels,
                                        kernel_size=net.conv1.kernel_size,
                                        stride=net.conv1.stride,
                                        padding=net.conv1.padding,
                                        bias=net.conv1.bias)

    def forward(self, observations):
        resnet = self._net
        device = resnet.conv1.weight.device

        # -- RGB preprocess
        rgb_observations = observations["rgb"].to(device).float()
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB

        x = resnet.conv1(rgb_observations)

        if self.use_depth:
            # -- Depth preprocess
            depth_observations = observations["depth"].to(device).float()
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            out_depth = self.depth_conv(depth_observations)
            x += out_depth

        # Continue with resnet
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)

        x = resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ReachabilityNet(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        self.extractor = nn.Sequential(
            nn.Linear(observation_space * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 2)
        )

    def forward(self, o1, o2):
        x = torch.cat([o1, o2], dim=1)
        x = self.extractor(x)

        return x


