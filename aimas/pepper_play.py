
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
observations = \
    pepper_env.reset()
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
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = observations['depth']
    if('position' in observations):
        pose = observations['position']
        print(pose)
        x_p.append(pose[0][0])
        y_p.append(pose[0][1])
        plt.clf()
        plt.plot(x_p, y_p)
        plt.plot(x_o, y_o, "--")
        plt.pause(0.01)

    print(i)
    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    # print('-'*100)
    # print(action)
    # print(pose)
    # print(rel_pose)
    #
    key = cv2.waitKey(0)
    i = i + 1
    if done:
        observations = \
            pepper_env.reset()

pepper_env.close()
