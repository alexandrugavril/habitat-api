import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from habitat_baselines.common.utils import Flatten


class SimpleCNNResnet(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, pretrained=False):
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

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
            self.cnn = nn.Sequential()
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

            self._net = net = resnet18(pretrained=pretrained, norm_layer=nn.InstanceNorm2d)

            self.use_depth = False
            if "depth" in observation_space.spaces:
                _n_depth = observation_space.spaces["depth"].shape[2]
                self.use_depth = True
                self.depth_conv = nn.Conv2d(_n_depth, net.conv1.out_channels,
                                            kernel_size=net.conv1.kernel_size,
                                            stride=net.conv1.stride,
                                            padding=net.conv1.padding,
                                            bias=net.conv1.bias)

            self.head = nn.Sequential(
                nn.Linear(512, output_size),  # Hardcoded input size from
                # resnet 18
                nn.ReLU(True),
            )

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
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        resnet = self._net

        rgb_observations = observations["rgb"]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        rgb_observations = rgb_observations / 255.0  # normalize RGB

        x = resnet.conv1(rgb_observations)

        if self.use_depth:
            # -- Depth preprocess
            depth_observations = observations["depth"].float()
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
        x = self.head(x)

        return x
