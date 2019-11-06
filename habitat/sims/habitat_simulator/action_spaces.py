from enum import Enum

import attr

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import (
    ActionSpaceConfiguration,
    Config,
    SimulatorActions,
)

from habitat_sim.agent.controls.pyrobot_noisy_controls import pyrobot_noise_models

@registry.register_action_space_configuration(name="v0")
class HabitatSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            SimulatorActions.STOP: habitat_sim.ActionSpec("stop"),
            SimulatorActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            SimulatorActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            SimulatorActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="v1")
class HabitatSimV1ActionSpaceConfiguration(
    HabitatSimV0ActionSpaceConfiguration
):
    def get(self):
        config = super().get()
        new_config = {
            SimulatorActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            SimulatorActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
        }

        config.update(new_config)

        return config


@registry.register_action_space_configuration(name="v2")
class HabitatSimV2ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def get(self):
        config = super().get()


        new_config = {
            SimulatorActions.NOISY_MOVE_FORWARD: habitat_sim.ActionSpec(
                "pyrobot_noisy_move_forward",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE,
                    robot=self.config.ROBOT,
                    controller=self.config.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MULTIPLIER,
                ),
            ),
            SimulatorActions.NOISY_TURN_LEFT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_left",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.ROBOT,
                    controller=self.config.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MULTIPLIER,
                ),
            ),
            SimulatorActions.NOISY_TURN_RIGHT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_right",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.ROBOT,
                    controller=self.config.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MULTIPLIER,
                ),
            ),
        }

        config.update(new_config)

        return config
