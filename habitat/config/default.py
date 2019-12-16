#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from habitat.config import Config as CN  # type: ignore

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT = -1

_C.ENVIRONMENT.OVERRIDE_RAND_GOAL = CN()
_C.ENVIRONMENT.OVERRIDE_RAND_GOAL.ENABLED = False
_C.ENVIRONMENT.OVERRIDE_RAND_GOAL.MIN_DIST = 1.
_C.ENVIRONMENT.OVERRIDE_RAND_GOAL.RADIUS = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # NAVIGATION TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SUCCESS_DISTANCE = 0.2
_C.TASK.SUCCESS_IF_IN_VIEW = True
_C.TASK.VIEW_FIELD_FACTOR = 0.2
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
_C.TASK.GOAL_SENSOR_UUID = "pointgoal"
_C.TASK.WITH_TARGET_ENCODING = True
_C.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
                            "LOOK_UP", "LOOK_DOWN"]
# -----------------------------------------------------------------------------
# # ACTIONS
# -----------------------------------------------------------------------------
ACTIONS = CN()
ACTIONS.STOP = CN()
ACTIONS.STOP.TYPE = "StopAction"
# -----------------------------------------------------------------------------
# # NAVIGATION ACTIONS
# -----------------------------------------------------------------------------
ACTIONS.MOVE_FORWARD = CN()
ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
ACTIONS.TURN_LEFT = CN()
ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
ACTIONS.TURN_RIGHT = CN()
ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
ACTIONS.LOOK_UP = CN()
ACTIONS.LOOK_UP.TYPE = "LookUpAction"
ACTIONS.LOOK_DOWN = CN()
ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
ACTIONS.TELEPORT = CN()
ACTIONS.TELEPORT.TYPE = "TeleportAction"

ACTIONS.NOISY_MOVE_FORWARD = CN()
ACTIONS.NOISY_MOVE_FORWARD.TYPE = "NoisyMoveForwardAction"
ACTIONS.NOISY_TURN_LEFT = CN()
ACTIONS.NOISY_TURN_LEFT.TYPE = "NoisyTurnLeftAction"
ACTIONS.NOISY_TURN_RIGHT = CN()
ACTIONS.NOISY_TURN_RIGHT.TYPE = "NoisyTurnRightAction"

_C.TASK.ACTIONS = ACTIONS
# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# #  SET GOAL
# -----------------------------------------------------------------------------
_C.TASK.SET_GOAL = CN()
_C.TASK.SET_GOAL.TYPE = "SetGoal"
_C.TASK.SET_GOAL.GOAL_SELECTION = "RANDOM"
# -----------------------------------------------------------------------------
# #  GET GOAL CLASS ONE HOT
# -----------------------------------------------------------------------------
_C.TASK.GET_GOAL_CLASS = CN()
_C.TASK.GET_GOAL_CLASS.TYPE = "GoalClass"
# -----------------------------------------------------------------------------
# #  MULTI POINT GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTMULTIGOAL_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTMULTIGOAL_SENSOR.TYPE = (
    "PointMultiGoalSensor"
)
# -----------------------------------------------------------------------------
# # POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.OBJECTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = "ObjectGoalWithGPSCompassSensor"
_C.TASK.OBJECTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# # POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GOAL_COORD_IN_CAMERA = CN()
_C.TASK.GOAL_COORD_IN_CAMERA.TYPE = "GoalCoordInCamera"
_C.TASK.GOAL_COORD_IN_CAMERA.HFOV = 90
# -----------------------------------------------------------------------------
# # POINTGOAL WITH GPS+COMPASS SENSOR+Object Scale
# -----------------------------------------------------------------------------
_C.TASK.GOAL_BBOX_IN_CAMERA = CN()
_C.TASK.GOAL_BBOX_IN_CAMERA.TYPE = "GoalBBoxInCamera"
_C.TASK.GOAL_BBOX_IN_CAMERA.HFOV = 90

# -----------------------------------------------------------------------------
# # POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.AGENT_POS_SENSOR = CN()
_C.TASK.AGENT_POS_SENSOR.TYPE = "AgentPosSensor"
# ----------------  -------------------------------------------------------------
# # HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# # COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COMPASS_SENSOR = CN()
_C.TASK.COMPASS_SENSOR.TYPE = "CompassSensor"
# -----------------------------------------------------------------------------
# # GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_SENSOR = CN()
_C.TASK.GPS_SENSOR.TYPE = "GPSSensor"
_C.TASK.GPS_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # GPS COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_COMPASS_SENSOR = CN()
_C.TASK.GPS_COMPASS_SENSOR.TYPE = "GPSCompassSensor"
_C.TASK.GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# # EMPTY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.EMPTY_SENSOR = CN()
_C.TASK.EMPTY_SENSOR.TYPE = "EmptySensor"
_C.TASK.EMPTY_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # GPS COMPASS START SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_COMPASS_START_SENSOR = CN()
_C.TASK.GPS_COMPASS_START_SENSOR.TYPE = "GPSCompassStartSensor"
_C.TASK.GPS_COMPASS_START_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# # GPS COMPASS Relative START SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_COMPASS_RELATIVE_SENSOR = CN()
_C.TASK.GPS_COMPASS_RELATIVE_SENSOR.TYPE = "GPSCompassRelativeSensor"
_C.TASK.GPS_COMPASS_RELATIVE_SENSOR.DIMENSIONALITY = 3
_C.TASK.GPS_COMPASS_RELATIVE_SENSOR.RELATIVE_STEP = 5
# -----------------------------------------------------------------------------
# # PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# # SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
_C.TASK.SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# # SPLMultiGoal MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPLMultiGoal = CN()
_C.TASK.SPLMultiGoal.TYPE = "SPLMultiGoal"
_C.TASK.SPLMultiGoal.SUCCESS_DISTANCE = 0.0
_C.TASK.SPLMultiGoal.SUCCESS_IF_IN_VIEW = True
_C.TASK.SPLMultiGoal.VIEW_FIELD_FACTOR = 0.2
# -----------------------------------------------------------------------------
# # TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP.DRAW_ALL_GOALS = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
# -----------------------------------------------------------------------------
# # COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.ROBOT = "LoCoBot"  # in metres
_C.SIMULATOR.CONTROLLER = "ILQR"  # in metres
_C.SIMULATOR.NOISE_MULTIPLIER = 1.0  # in metres
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/" "van-gogh-room.glb"
)
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
# -----------------------------------------------------------------------------
# # SENSORS
# -----------------------------------------------------------------------------
SENSOR = CN()
SENSOR.HEIGHT = 480
SENSOR.WIDTH = 640
SENSOR.HFOV = 90  # horizontal field of view in degrees
SENSOR.POSITION = [0, 1.25, 0]
SENSOR.V_OFFSET_NOISE = 0
SENSOR.V_OFFSET_NUM_STEPS = 100
SENSOR.BATCH = 1
# -----------------------------------------------------------------------------
# # RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
_C.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.
_C.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.
_C.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
# DEPTH SENSOR 2
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH2_SENSOR = SENSOR.clone()
_C.SIMULATOR.DEPTH2_SENSOR.TYPE = "HabitatSimDepthSensor2"
_C.SIMULATOR.DEPTH2_SENSOR.MIN_DEPTH = 0.
_C.SIMULATOR.DEPTH2_SENSOR.MAX_DEPTH = 10.
_C.SIMULATOR.DEPTH2_SENSOR.NORMALIZE_DEPTH = True
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.MASS = 32.0
_C.SIMULATOR.AGENT_0.LINEAR_ACCELERATION = 20.0
_C.SIMULATOR.AGENT_0.ANGULAR_ACCELERATION = 4 * 3.14
_C.SIMULATOR.AGENT_0.LINEAR_FRICTION = 0.5
_C.SIMULATOR.AGENT_0.ANGULAR_FRICTION = 1.0
_C.SIMULATOR.AGENT_0.COEFFICIENT_OF_RESTITUTION = 0.0
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
