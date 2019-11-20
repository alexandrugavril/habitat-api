
from habitat_baselines.config.default import get_config
from habitat_baselines.common.pepper_env import PepperRLExplorationEnv
import cv2
import pickle
import numpy as np

rgb_buffer = []
depth_buffer = []
forward_step = 0.25
turn_step = 0.1

cfg = get_config()
print(cfg)
pepper_env = PepperRLExplorationEnv(cfg)
pepper_env.reset()

key = 0
c_action = np.random.choice(3, 1, p=[0.8, 0.1, 0.1])[0]
default_rot = np.random.choice(3, 1, p=[0, 0.5, 0.5])[0]

values = []
forward_enabled = True
user_forward_enabled = True
num_forward = 0

actions = []
step = 0
observations, reward, done, info = \
    pepper_env.reset()
last_pose = pepper_env.get_position()[0]

while key != ord('q'):
    step += 1
    last_action = c_action
    observations, reward, done, info = \
        pepper_env.step(None, action={"action": c_action})
    pose = pepper_env.get_position()
    sonar = pepper_env.get_sonar()

    movement = np.linalg.norm(pose[0] - last_pose)

    print("-" * 100)
    print("Sonar:", sonar)
    print("Pose:", pose)
    print("Movement:", movement)
    print("STEP:", step)
    rgb = observations['rgb']
    depth = observations['depth']

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    key = cv2.waitKey(500)

    if sonar < 0.9 or (last_action == 0 and movement < 0.15):
        forward_enabled = False
        print("Disabled forward")
    else:
        forward_enabled = True
        print("Enabled forward")

    if forward_enabled and user_forward_enabled:
        c_action = np.random.choice(3, 1, p=[0.8, 0.1, 0.1])[0]
    else:
        print("Forward is disabled")
        c_action = default_rot
        print("Running with default rotation")

    if c_action == 0:
        num_forward += 1
    else:
        num_forward = 0

    if num_forward == 3:
        print("Changing default rotation")
        default_rot = np.random.choice(3, 1, p=[0, 0.5, 0.5])[0]
        num_forward = 0

    if key == ord('w'):
        user_forward_enabled = not user_forward_enabled
    #elif key == ord('a'):
    #    c_action = 1
    #elif key == ord('d'):
    #    c_action = 2

    values.append({
        "rgb": rgb,
        "depth": depth,
        "position": pose[0],
        "rotation": pose[1],
        "action": last_action,
        "sonar": sonar
    })
    last_pose = pose[0]
    if step == 250:
        break

import datetime
now = datetime.datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
pickle.dump(values, open(dt_string + "pepper_save.p", "wb"))
pepper_env.close()
