#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.MULTIPLY_SCENES = False
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VIDEO_OPTION_INTERVAL = 10
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 5000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.COMMIT = ""
_C.EVAL_MODE = False

# -----------------------------------------------------------------------------
# IMAGE TRANSFORM
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()
_C.TRANSFORM.ENABLED = False
_C.TRANSFORM.min_scale = 1.
_C.TRANSFORM.max_scale = 1.2

_C.TRANSFORM_RGB = CN()
_C.TRANSFORM_RGB.ENABLED = False

_C.TRANSFORM_DEPTH = CN()
_C.TRANSFORM_DEPTH.ENABLED = False
_C.TRANSFORM_DEPTH.std_noise = 0.2
_C.TRANSFORM_DEPTH.depth_min_th = 0.04
_C.TRANSFORM_DEPTH.depth_max_th = 0.8
_C.TRANSFORM_DEPTH.noise_interval = 0.03

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.COLLISION_REWARD_ENABLED = False
_C.RL.COLLISION_DISTANCE = 0.
_C.RL.COLLISION_REWARD = -1.

# -----------------------------------------------------------------------------
# DETECTOR
_C.DETECTOR = CN()
_C.DETECTOR.model_def = "yolov3/config/yolov3.cfg"
_C.DETECTOR.weights_path = "yolov3/weights/yolov3.weights"
_C.DETECTOR.class_path = "yolov3/data/coco.names"
_C.DETECTOR.iou_thres = 0.5
_C.DETECTOR.conf_thres = 0.2
_C.DETECTOR.img_size = 256
_C.DETECTOR.nms_thres = 0.4
_C.DETECTOR.out_size = 32


# -----------------------------------------------------------------------------
# PEPPER
# -----------------------------------------------------------------------------
_C.PEPPER = CN()
_C.PEPPER.BufferSize = 10
_C.PEPPER.ForwardStep = 0.25
_C.PEPPER.TurnStep = 0.08725
_C.PEPPER.RGBTopic = "/pepper_robot/naoqi_driver/camera/front/image_raw"
_C.PEPPER.DepthTopic = "/pepper_robot/naoqi_driver/camera/depth/image_raw"
_C.PEPPER.MoveTopic = "/move_base_simple/goal"
_C.PEPPER.PoseTopic = "/slam_out_pose"
_C.PEPPER.GoalTopic = "/clicked_point"
_C.PEPPER.SonarTopic = "/pepper_robot/naoqi_driver/sonar/front"

_C.PEPPER.DisplayImages = False
_C.PEPPER.EpisodePath = "19.11.2019 11:46:14pepper_save.p"

# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.visual_encoder = "SimpleCNN"
_C.RL.PPO.visual_encoder_dropout = 0.0
_C.RL.PPO.channel_scale = 1
_C.RL.PPO.clip_param = 0.1
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 4
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 256
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = True
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
# -----------------------------------------------------------------------------
# REACHABILITY NETWORK CONFIG
# -----------------------------------------------------------------------------
_C.RL.REACHABILITY = CN()
_C.RL.REACHABILITY.enabled = False
_C.RL.REACHABILITY.train = False
_C.RL.REACHABILITY.skip_train_ppo_without_rtrain = False
_C.RL.REACHABILITY.only_intrinsic_reward = False
_C.RL.REACHABILITY.experience_buffer_size = 720000
_C.RL.REACHABILITY.num_recurrent_steps = 1
_C.RL.REACHABILITY.batch_size = 64
_C.RL.REACHABILITY.num_train_epochs = 10
_C.RL.REACHABILITY.feature_extractor_size = 512
_C.RL.REACHABILITY.memory_size = 200
_C.RL.REACHABILITY.log_freq = 10
_C.RL.REACHABILITY.grid_resolution = 1

_C.RL.REACHABILITY.optimizer = "Adam"
_C.RL.REACHABILITY.optimizer_args = CN()
_C.RL.REACHABILITY.optimizer_args.lr = 0.00025

# Hyperparams
_C.RL.REACHABILITY.max_action_distance_k = 5
_C.RL.REACHABILITY.negative_sample_multiplier = 5
_C.RL.REACHABILITY.curiosity_bonus_scale_a = 0.030
_C.RL.REACHABILITY.reward_shift_b = 0.5
_C.RL.REACHABILITY.novelty_threshold = 0.5
_C.RL.REACHABILITY.similarity_aggregation = "percentile"
_C.RL.REACHABILITY.similarity_percentile = 90


# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()

    task_config_opts = []

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:

            # Read config and extract TASK_CONFIG args to add as opts
            with open(config_path, "r") as f:
                cfg = config.load_cfg(f)

            cfg_clone = cfg.clone()
            for k in list(cfg.keys()):
                if not k.startswith("TASK_CONFIG"):
                    cfg_clone.pop(k)
                else:
                    cfg.pop(k)

            task_config_opts.append(cfg_clone)

            config.merge_from_other_cfg(cfg)

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    for tsk_opts in task_config_opts:
        config.merge_from_other_cfg(tsk_opts)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
