#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import pandas as pd

import numpy as np
import pytest

from typing import Optional
from typing import List
import habitat
from habitat.core.simulator import Simulator
from habitat.datasets.pointnav.pointnav_generator import \
    ISLAND_RADIUS_LIMIT, is_compatible_episode
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.pointnav.pointnav_dataset import (
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav_task import (
    NavigationGoal,
    NavigationEpisode,
)
from habitat.utils.geometry_utils import quaternion_xyzw_to_wxyz
from habitat.core.simulator import ShortestPathPoint, SimulatorActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_to_list


BASE_CFG_PATH = "configs/tasks/pointnav.yaml"
SCENE_INFO_PATH = "../Replica-Dataset/dataset/semantic_data.npy"
SCENE_MOCKUP_PATH = "../Replica-Dataset/dataset/{}/habitat/mesh_semantic.ply"
NUM_EPISODES = 100

# FILTER OBJECTS
MATCHING_CLASSES = dict({  # REPLICA class name: COCO class name
    "book": "book",
    "chair": "chair",
    "table": "diningtable",
    "bowl": "bowl",
    "bottle": "bottle",
    "indoor-plant": "pottedplant",
    "cup": "cup",
    "vase": "vase",
    "tv-screen": "tvmonitor",
    "sofa": "sofa",
    "bike": "bicycle",
    "sink": "sink"
})


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    shortest_paths=None,
    radius=None,
    info=None,
    multi_target=None,
    multi_taget_radius=0.2,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position, radius=radius)]
    if multi_target is not None:
        for t in multi_target:
            goals.append(NavigationGoal(position=t, radius=multi_taget_radius))

    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def get_action_shortest_path(
    sim,
    source_position,
    source_rotation,
    goal_position,
    success_distance=0.05,
    max_episode_steps=500,
    shortest_path_mode="geodesic_path",
    max_still_moves= 5,
) -> List[ShortestPathPoint]:
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    follower = ShortestPathFollower(sim, success_distance, False)
    follower.mode = shortest_path_mode

    shortest_path = []
    step_count = 0
    action = follower.get_next_action(goal_position)

    state = sim.get_agent_state()
    prev_pos, prev_rot = state.position, state.rotation
    no_move = 0

    while action is not None and step_count < max_episode_steps:
        state = sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action,
            )
        )
        sim.step(action)
        step_count += 1
        action = follower.get_next_action(goal_position)
        if (prev_pos == state.position).all() and \
            prev_rot == state.rotation:
            no_move += 1
            if no_move > max_still_moves:
                logger.warning(f"Reached {no_move} steps X no movement.")
                break
        else:
            no_move = 0
            prev_pos, prev_rot = state.position, state.rotation

    if step_count == max_episode_steps:
        logger.warning("Shortest path wasn't found.")
    return shortest_path


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def _random_episode(env, config):
    random_location = env._sim.sample_navigable_point()
    random_heading = np.random.uniform(-np.pi, np.pi)
    random_rotation = [
        0,
        np.sin(random_heading / 2),
        0,
        np.cos(random_heading / 2),
    ]
    env.episode_iterator = iter(
        [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=random_location,
                start_rotation=random_rotation,
                goals=[NavigationGoal(position=random_location)],
                info={"geodesic_distance": 0.001},
            )
        ]
    )


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = str(dataset.to_json())
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = dataset.__class__()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) > 0
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_single_pointnav_dataset():
    dataset_config = get_config().DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) == 0
    ), "Expected dataset doesn't expect separate episode file per scene."
    dataset = PointNavDatasetV1(config=dataset_config)
    assert len(dataset.episodes) > 0, "The dataset shouldn't be empty."
    assert (
        len(dataset.scene_ids) == 2
    ), "The test dataset scenes number is wrong."
    check_json_serializaiton(dataset)


def test_multiple_files_scene_path(cfg_path: str):
    dataset_config = get_config(cfg_path).DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    dataset_config.defrost()
    dataset_config.CONTENT_SCENES = scenes
    dataset_config.SCENES_DIR = os.path.join(
        os.getcwd(), DEFAULT_SCENE_PATH_PREFIX
    )
    dataset_config.freeze()
    partial_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    print(partial_dataset.episodes[0].scene_id)
    assert os.path.exists(
        partial_dataset.episodes[0].scene_id
    ), "Scene file {} doesn't exist using absolute path".format(
        partial_dataset.episodes[0].scene_id
    )


def test_multiple_files_pointnav_dataset(cfg_path: str):
    dataset_config = get_config(cfg_path).DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    dataset_config.defrost()
    dataset_config.CONTENT_SCENES = scenes
    dataset_config.freeze()
    partial_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    check_json_serializaiton(partial_dataset)


def check_shortest_path(env, episode):
    def check_state(agent_state, position, rotation):
        assert np.allclose(
            agent_state.rotation, quaternion_xyzw_to_wxyz(rotation)
        ), "Agent's rotation diverges from the shortest path."

        assert np.allclose(
            agent_state.position, position
        ), "Agent's position position diverges from the shortest path's one."

    # assert len(episode.goals) == 1, "Episode has no goals or more than one."
    assert (
        len(episode.shortest_paths) == 1
    ), "Episode has no shortest paths or more than one."

    env.episode_iterator = iter([episode])
    env.reset()
    start_state = env.sim.get_agent_state()
    check_state(start_state, episode.start_position, episode.start_rotation)

    for step_id, point in enumerate(episode.shortest_paths[0]):
        cur_state = env.sim.get_agent_state()
        check_state(cur_state, point.position, point.rotation)
        observations = env.step(point.action)


def is_compatible_episode(
    s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 3.0:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, t)
    if d_separation == np.inf:
        return False, 1
    if not near_dist <= d_separation <= far_dist:
        return False, 2
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 3
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 4
    return True, d_separation


def sample_navigable_point(sim: Simulator, floor_coord: List[float],
                           max_dist: float = 0.3, max_tries: float = 100):
    """ Get sample of point only on ground """
    for _ in range(max_tries):
        source_position = sim.sample_navigable_point()

        dist = source_position[1] - floor_coord
        if np.min(np.abs(dist)) < max_dist:  # TODO Fix points on ceileing
            return source_position
    return None


def generate_pointnav_episode(
    sim: Simulator,
    target: List = None,  # Specify target Coord
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    geodesic_min_ratio_prob: float = 0.01,
    number_retries_per_target: int = 50,
    floor_coord: List[float] = None,
    max_samples_multi_target: int = 40,
) -> NavigationEpisode:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    multiple_targets = None
    target_position = None
    floor_coord = np.array(floor_coord)
    target_idx = 0

    while episode_count < num_episodes or num_episodes < 0:
        if target_position is None:
            target_position = sim.sample_navigable_point() if target is None \
                else target

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT and \
            target is None:
            continue

        if multiple_targets is not None:
            target_idx = np.random.randint(0, len(multiple_targets))
            target = multiple_targets[target_idx]
            target_idx += 1

        found_episode = False
        episode = None
        for retry in range(number_retries_per_target):
            source_position = sample_navigable_point(sim, floor_coord)

            min_r = 0 if np.random.rand() < geodesic_min_ratio_prob else \
                geodesic_to_euclid_min_ratio

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=min_r,
            )

            if is_compatible:
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                shortest_paths = None
                if is_gen_shortest_path:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=sim.config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    target_position=target_position,
                    shortest_paths=shortest_paths,
                    radius=shortest_path_success_distance,
                    info={"geodesic_distance": dist},
                    multi_target=multiple_targets
                )
                episode.target_idx = target_idx

                episode_count += 1
                found_episode = True
                break

        if not found_episode:
            print("Can't reach actual object coordinate, try to find "
                  "shortest reachable points")

            min_dist = 0.2
            end = []
            while len(end) < max_samples_multi_target:
                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                source_position = sample_navigable_point(sim, floor_coord)

                shp = get_action_shortest_path(
                    sim,
                    source_position=source_position,
                    source_rotation=source_rotation,
                    goal_position=target_position,
                    success_distance=min_dist,
                    max_episode_steps=shortest_path_max_steps,
                )
                if len(shp) <= 0:
                    continue

                shp = np.array([x.position for x in shp])
                eucl_distance = \
                    np.linalg.norm(shp - np.array(target), axis=1)

                end.append(shp[eucl_distance.argmin()])

            end = np.array(end)

            eucl_distance = \
                np.linalg.norm(end - np.array(target), axis=1)
            sortidx = np.argsort(eucl_distance)
            accepted_distance = np.mean(eucl_distance[sortidx][:3]) + min_dist
            accepted_distance = max(accepted_distance,
                                    shortest_path_success_distance)

            multiple_targets = end[
                eucl_distance < accepted_distance]

            multiple_targets = np.unique(multiple_targets, axis=0)
            print(multiple_targets)
            assert len(multiple_targets) > 0, "Still no targets"

            continue

        yield episode


def determine_height_to_object(obj_coord, obj_size, floor_coord: List[float]):
    h_axis = 1
    obj_h, obj_sh = obj_coord[h_axis], obj_size[h_axis]

    dist_to_floor = np.where(obj_h > floor_coord)[0]
    if len(dist_to_floor) < 1:
        return None

    floor_idx = dist_to_floor[-1]
    floor_h = floor_coord[floor_idx]
    return obj_h - floor_h - obj_sh//2


def generate_episodes():
    # -- Load scene files
    semantic_data = np.load(SCENE_INFO_PATH, allow_pickle=True).item()
    df_semantic = semantic_data["df_semantic"]
    df_objects = semantic_data["df_objects"]

    # filter classes
    selected_classes = MATCHING_CLASSES.keys()
    df_objects = df_objects[df_objects.class_name.apply(lambda x: x in selected_classes)]
    print("_____________CLASSES INFO_____________")
    print(df_objects.class_name.value_counts())
    print(f"Total count: {len(df_objects)}")
    print("_______________________________________")

    # -- Load config
    config = get_config(BASE_CFG_PATH)
    base_success = config.TASK.SUCCESS_DISTANCE

    env = habitat.Env(config)
    env.seed(config.SEED)
    random.seed(config.SEED)

    dataset_episodes = []
    for room in df_objects.room.unique():
        print(f"Generating episodes for room {room}")
        if room in ["office_1", "office_2", "room_2", "frl_apartment_0",
                    "office_3"]:
            continue

        scene_path = SCENE_MOCKUP_PATH.format(room)
        scene_df = df_objects[df_objects.room == room]

        config.defrost()
        config.SIMULATOR.SCENE = scene_path
        config.freeze()
        _random_episode(env, config)
        out = env.reset()

        # =====================================================================
        # Determine floor coord/s

        pts = []
        while len(pts) < 300:
            pt = env.sim.sample_navigable_point()
            if env.sim.island_radius(pt) > ISLAND_RADIUS_LIMIT:
                pts.append(pt[1])
        floor_coords = pd.value_counts(pts).index.values
        floor_coord = []
        while len(floor_coords) > 0:
            floor_coord.append(floor_coords[0])
            floor_coords = floor_coords[floor_coords > (floor_coord[-1] + 2.3)]

        print(f"Room {room} has {len(floor_coord)} floors @ ({floor_coord})")
        print("objects", scene_df.class_name.unique())
        # =====================================================================

        for obj_idx, obj_row in scene_df.iterrows():
            print(f"-generating {NUM_EPISODES} for {obj_row['class_name']}")

            t_coord = obj_row["habitat_coord"]
            t_size = obj_row["habitat_size"]

            # Determine success distance
            h_to_obj = determine_height_to_object(t_coord, t_size, floor_coord)
            if h_to_obj is None:
                print(f"Coord under floor {obj_idx}, coord: {t_coord}")
                continue

            success_distance = \
                determine_height_to_object(t_coord, t_size, floor_coord) + \
                base_success + max(t_size) / 2.

            generator = generate_pointnav_episode(
                sim=env.sim,
                target=t_coord,
                shortest_path_success_distance=success_distance,
                shortest_path_max_steps=config.ENVIRONMENT.MAX_EPISODE_STEPS,
                geodesic_to_euclid_min_ratio=1.1,
                number_retries_per_target=50,
                geodesic_min_ratio_prob=0.5,
                floor_coord=floor_coord,
            )

            episodes = []

            for i in range(NUM_EPISODES):
                episode = next(generator)

                # Add arguments
                episode.room = room
                episode.t_coord = t_coord
                episode.t_size = t_size
                episode.class_name = obj_row['class_name']

                episodes.append(episode)

            for episode in episodes:
                check_shortest_path(env, episode)

            dataset_episodes += episodes

        np.save("dataset_all", dataset_episodes)

    dataset = habitat.Dataset()
    dataset.episodes = dataset_episodes
    # assert dataset.to_json(), "Generated episodes aren't json serializable."


if __name__ == "__main__":
    generate_episodes()
