
from habitat_baselines.config.default import get_config
from habitat_baselines.common.pepper_playback_env import PepperPlaybackEnv
import cv2
import pickle

rgb_buffer = []
depth_buffer = []
forward_step = 0.25
turn_step = 0.1

cfg = get_config(config_paths=['./results/2019-11-05_20-19-38_new_param_noise/'
                               'ppo_explore_GO_replica.yaml'])
print(cfg)
pepper_env = PepperPlaybackEnv(cfg)

key = 0
while key != ord('q'):
    observations, reward, done, info = \
        pepper_env.step(None, action={"action": 0})

    rgb = observations['rgb']
    depth = observations['depth']
    pose = info['position']
    action = info['action']

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    print(pose)
    print(action)
    key = cv2.waitKey(0)
    if done:
        break

pepper_env.close()
