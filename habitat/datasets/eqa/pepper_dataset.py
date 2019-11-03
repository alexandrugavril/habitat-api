#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.dataset import Dataset

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PepperRobot")
class PepperRobot(Dataset):

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r""" Dummy list of scenes
        """
        return [None]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config
        self.episodes = []
