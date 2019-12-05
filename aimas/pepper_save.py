
from habitat_baselines.config.default import get_config
from habitat_baselines.common.pepper_env import PepperRLExplorationEnv
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
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

x_p = []
y_p = []

x_o = []
y_o = []


plt.ion()
plt.show()

while key != ord('q'):
    step += 1
    last_action = c_action
    observations, reward, done, info = \
        pepper_env.step(None, action={"action": c_action})
    pose = observations['robot_position']
    rot = observations['robot_rotation']
    sonar = observations['sonar']
    odom = observations['odom']


    gps_to_goal = observations['gps_with_pointgoal_compass']
    movement = np.linalg.norm(pose - last_pose)

    print("-" * 100)
    print("Pose:", pose)
    print("Odom:", odom)
    print("Sonar:", sonar)
    print("Movement:", movement)
    print("STEP:", step)

    x_p.append(pose[0])
    y_p.append(pose[1])
    x_o.append(odom[0][0])
    y_o.append(odom[0][1])

    plt.clf()
    plt.plot(x_p, y_p)
    plt.plot(x_o, y_o, "--")
    plt.pause(0.01)

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
        "odom_pose": odom[0],
        "odom_rot": odom[1],
        "position": pose,
        "rotation": rot,
        "action": last_action,
        "gps_to_goal_compass": gps_to_goal,
        "sonar": sonar
    })
    last_pose = pose
    if step == 250:
        break

import datetime
now = datetime.datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
pickle.dump(values, open(dt_string + "pepper_save.p", "wb"))
pepper_env.close()

