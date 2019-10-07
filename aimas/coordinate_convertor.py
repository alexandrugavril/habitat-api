dataset_path = "/raid/workspace/alexandrug/Replica-Dataset/dataset"
import habitat
import cv2
from aimas.generate_episodes import dataset
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
import numpy as np
import quaternion


def change_pos(obj_pos, obj_rot):
    rotation_habitat_replica = quat_from_two_vectors(np.array([0, 0, -1]), np.array([0, -1, 0]))
    rotation_replica_habitat = rotation_habitat_replica.inverse()

    obj_quat = quaternion.from_float_array(obj_rot).inverse()

    obj_pos = quat_rotate_vector(rotation_replica_habitat, obj_pos)
    obj_position_replica = quat_rotate_vector(obj_quat, obj_pos)

    print(obj_position_replica)
    return obj_position_replica


fridge_pos = [4.3967685699462891,
              -0.63215178251266479,
              -0.55637705326080322]

fridge_rot = [0.99986499547958374,
                -0.00077493180288001895,
              0.00056659855181351304,
              0.016408002004027367,
              ]

agent_fridge_pos = [4.5105104, -1.3150995, 0.50484991]

umbrella_pos = [6.0712509155273438,
                -5.0927877426147461,
                -1.0105148553848267]

umbrella_rot = [0.7106926441192627,
                -0.00016766691987868398,
                0.00094522000290453434,
                0.70350199937820435,
                ]

agent_umbrella_pos = [4.9492435, -1.3150995, -5.8451505]



change_pos(fridge_pos, fridge_rot)
print(agent_fridge_pos)
print()
change_pos(umbrella_pos, umbrella_rot)
print(agent_umbrella_pos)
