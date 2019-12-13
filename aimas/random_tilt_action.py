import attr
import numpy as np

import habitat
import habitat_sim
from habitat.sims.habitat_simulator.action_spaces import (
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav_task import SimulatorAction


def _random_tilt_impl(
    scene_node: habitat_sim.SceneNode,
    move_amount: float,
    angle: float,
    noise_amount: float,
):
    scene_node.reset_transformation()
    scene_node.rotate_x_local(scene_node, angle, 0)

@attr.s(auto_attribs=True, slots=True)
class RandomTiltActuationSpec:
    move_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    strafe_angle: float = 90.0
    noise_amount: float = 0.05

@habitat_sim.registry.register_move_fn(body_action=False)
class RandomTilt(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: RandomTiltActuationSpec,
    ):
        print(f"strafing left with noise_amount={actuation_spec.noise_amount}")
        _random_tilt_impl(
            scene_node,
            actuation_spec.move_amount,
            actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat.registry.register_action_space_configuration
class RandomTilt(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[habitat.SimulatorActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            RandomTiltActuationSpec(0.25, noise_amount=0.05),
        )
        config[habitat.SimulatorActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            RandomTiltActuationSpec(0.25, noise_amount=0.05),
        )
        return config
