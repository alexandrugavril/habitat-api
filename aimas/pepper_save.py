
from habitat_baselines.config.default import get_config
from habitat_baselines.common.pepper_env import PepperRLExplorationEnv
import cv2
import pickle

rgb_buffer = []
depth_buffer = []
forward_step = 0.25
turn_step = 0.1

cfg = get_config(config_paths=['./results/2019-11-05_20-19-38_new_param_noise/'
                               'ppo_explore_GO_replica.yaml'])
print(cfg)
pepper_env = PepperRLExplorationEnv(cfg)

key = 0
c_action = 1

values = []

while key != ord('q'):
    last_action = c_action
    observations, reward, done, info = \
        pepper_env.step(None, action={"action": c_action})
    pose = pepper_env.get_position()
    print(pose)

    rgb = observations['rgb']
    depth = observations['depth']

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    key = cv2.waitKey(1)
    if key == ord('w'):
        c_action = 0
    elif key == ord('a'):
        c_action = 1
    elif key == ord('d'):
        c_action = 2

    values.append({
        "rgb": rgb,
        "depth": depth,
        "position": pose[0],
        "rotation": pose[1],
        "action": last_action
    })

pickle.dump(values, open("pepper_save.p", "wb"))
pepper_env.close()
