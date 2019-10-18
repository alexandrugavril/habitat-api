#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav_task import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PointNav-v2")
class PointNavDatasetV2(PointNavDatasetV1):

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert PointNavDatasetV1.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = PointNavDatasetV2(cfg)
        return PointNavDatasetV2._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    def __init__(self, config: Optional[Config] = None) -> None:
        self.navigation_base = [x.name for x in
                                NavigationEpisode.__attrs_attrs__]
        super(PointNavDatasetV2, self).__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        navigation_base = self.navigation_base

        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            other_keys = dict()
            for k, v in episode.items():
                if k not in navigation_base:
                    other_keys[k] = v
            for k in other_keys.keys():
                episode.pop(k)

            episode = NavigationEpisode(**episode)
            for k, v in other_keys.items():
                episode.__dict__[k] = v

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
