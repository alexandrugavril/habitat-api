
from habitat_baselines.config.default import get_config
from habitat_baselines.common.pepper_playback_env import PepperPlaybackEnv
import cv2
import pickle
import matplotlib.pyplot as plt

rgb_buffer = []
depth_buffer = []
forward_step = 0.25
turn_step = 0.1

cfg = get_config()
print(cfg)
pepper_env = PepperPlaybackEnv(cfg)
x_p = []
y_p = []
x_o = []
y_o = []
key = 0
i = 0
while key != ord('q'):
    observations, reward, done, info = \
        pepper_env.step(None, action={"action": 0})

    rgb = observations['rgb']
    depth = observations['depth']
    pose = observations['position']
    print(pose)
    rot = observations['rotation']
    action = observations['action']
    rel_pose = observations['gps_compass']

    print(i)

    x_p.append(pose[0][0])
    y_p.append(pose[0][1])

    plt.clf()
    plt.plot(x_p, y_p)
    plt.plot(x_o, y_o, "--")
    plt.pause(0.01)

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    # print('-'*100)
    # print(action)
    # print(pose)
    # print(rel_pose)
    #
    key = cv2.waitKey(1)
    i = i + 1
    if done:
        break

pepper_env.close()
