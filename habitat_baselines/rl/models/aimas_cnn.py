r"""
    Credits and part of script from:
    https://github.com/facebookresearch/habitat-api
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import Flatten


class AimasCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, detector):
        super().__init__()
        self.detector = detector

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        self._no_classes = observation_space.spaces["goalclass"].shape[0]
        self._detector_channels = 765 // (3 * 3)

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            self.cnn_1 = nn.Sequential()
            self.cnn_2 = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True)
            )

            self.detector_cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._detector_channels + self._no_classes,
                    out_channels=64,
                    kernel_size=1,
                    stride=1,
                ),
                nn.ReLU(True),
            )

            self.cnn_2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64 + 64,
                    out_channels=128,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=32,
                    kernel_size=1,
                    stride=1,
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(32 * (cnn_dims[0] + 2) * (cnn_dims[1] + 2),
                          output_size),
                nn.ReLU(True),
            )
        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in list(self.cnn_1) + list(self.cnn_2):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations, target_encoding):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        x = self.cnn_1(cnn_input)

        x = F.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value=0)

        if "detector_features" in observations:
            detections = observations["detector_features"]
        else:
            detections = self.detector.detect(rgb_observations)
            observations["detector_features"] = detections

        detections = detections.detach()

        # Add target_encoding
        b, c, w, h = detections.size()
        b, tc = target_encoding.size()
        target_encoding = target_encoding.view(b, tc, 1, 1)
        target_encoding = target_encoding.expand(b, tc, w, h)

        detections = torch.cat([detections, target_encoding], dim=1)
        detections = self.detector_cnn(detections)

        x = torch.cat([detections, x], dim=1)

        x = self.cnn_2(x)

        return x
