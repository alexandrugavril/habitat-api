#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
import cv2

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.aimas_cnn import AimasCNN

from habitat_baselines.rl.ppo.policy import Policy, Net
from habitat.tasks.nav.nav_task_multi_goal import CLASSES

from yolov3.models import Darknet
from yolov3.utils import utils as yolo_utils


class ObjectClassNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        detector_config,
        device,
        hidden_size=512,
    ):
        super().__init__(
            ObjectClassNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                detector_config=detector_config,
                device=device
            ),
            action_space.n,
        )


class ObjectClassNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid,
                 detector_config, device):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size

        self.detector = detector = YoloDetector(detector_config, device)
        self.visual_encoder = AimasCNN(observation_space, hidden_size,
                                       detector)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
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
        target_encoding = self.get_target_encoding(observations)
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations,
                                                   target_encoding)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states, dict()


class YoloDetector:
    def __init__(self, config, device):
        self.opt = opt = config
        self.conf_thres = opt['conf_thres']
        self.nms_thres = opt['nms_thres']
        self.img_size = opt['img_size']
        self.out_img_size = out_size = opt['out_size']

        # Set up model
        self.model = Darknet(opt['model_def'], img_size=opt['img_size'])\
            .to(device)

        if opt['weights_path'].endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(opt['weights_path'])
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(opt['weights_path']))

        self.model.eval()  # Set in evaluation mode
        # Extracts class labels from file
        self.classes = yolo_utils.load_classes(opt['class_path'])

        mode = "nearest"
        self.b1_scale = nn.Upsample(scale_factor=out_size // 8, mode=mode)
        self.b2_scale = nn.Upsample(scale_factor=out_size // 16, mode=mode)
        self.b3_scale = nn.Upsample(scale_factor=out_size // 32, mode=mode)
        self.no_detects = 0

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape

        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x

        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes

    @staticmethod
    def class_selector():
        with open("yolov3/data/coco.names", "r") as f:
            yolo_classes = f.readlines()
            yolo_classes = [x.strip() for x in yolo_classes]

        indexer = torch.zeros(len(CLASSES), len(yolo_classes)).bool()

        for i, (k, v) in enumerate(CLASSES.items()):
            indexer[i, yolo_classes.index(v)] = 1
        return indexer

    def detect(self, rgb_img):
        self.no_detects += 1

        max_batch = 128

        """ Should run with RGB images normalized in [0, 1] """
        with torch.no_grad():
            multi_batch = []
            all_imgs = rgb_img
            for i in range(len(all_imgs) // max_batch + 1):
                rgb_img = all_imgs[i*max_batch: (i+1)*max_batch]
                if len(rgb_img) <= 0:
                    break

                bs = rgb_img.size(0)

                detections = self.model(rgb_img)
                b1, b2, b3 = (detections[:, :192], detections[:, 192: 960],
                              detections[:, 960:])

                ordd = (0, 1, 4, 2, 3)

                b1 = b1.view(bs, 3, 8, 8, 85).permute(*ordd).contiguous().view(bs, -1, 8, 8)
                b2 = b2.view(bs, 3, 16, 16, 85).permute(*ordd).contiguous().view(bs, -1, 16, 16)
                b3 = b3.view(bs, 3, 32, 32, 85).permute(*ordd).contiguous().view(bs, -1, 32, 32)

                b1 = self.b1_scale(b1)
                b2 = self.b2_scale(b2)
                b3 = self.b3_scale(b3)

                out = (b1 + b2 + b3) / 3
                out = out.view(bs, 3, 85, 32, 32)
                out = out.mean(dim=1)#.permute(0, 3, 1, 2)
                # out = torch.cat([b1, b2, b3], dim=1)
                multi_batch.append(out)

        if len(multi_batch) > 1:
            out = torch.cat(multi_batch, dim=0)
        else:
            out = multi_batch[0]

        out = out.detach()
        return out

    def get_bounding_boxes(self, img, display=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t_img = torch.from_numpy(
            img.astype('float') / 255.0).cuda().permute(2, 0, 1).unsqueeze(0)

        detections = self.model(t_img)
        detections = yolo_utils.non_max_suppression(detections,
                                                    self.conf_thres,
                                                    self.nms_thres)[0]

        # Draw bounding boxes and labels of detections
        if display:
            img_disp = img.copy()

        if detections is not None:
            # Rescale boxes to original image
            detections = self.rescale_boxes(detections, self.img_size,
                                            img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" %
                      (self.classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                print("{} {} {} {}" .format(x1, y1, x2, y2))

                # Create a Rectangle patch
                if display:
                    img_disp = cv2.rectangle(img_disp, (x2, y2), (x1, y1),
                                             (255,0,0), 2)

        if display:
            cv2.imshow("Test", img_disp)
            cv2.waitKey(0)

        return detections
