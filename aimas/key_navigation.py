import habitat
import cv2
import pandas as pd

from habitat.core.dataset import Dataset, Episode
from habitat.tasks.nav.nav_task import (
    NavigationEpisode,
    NavigationGoal,
    StopAction,
)
from habitat.utils.visualizations import maps

from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
import numpy as np
import quaternion
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
LOOK_UP = "8"
LOOK_DOWN = "2"


def get_goal(room, ep):
    DATA_PATH = "dataset_all.npy"
    data = np.load(DATA_PATH, allow_pickle=True)

    df = pd.DataFrame([x.__dict__ for x in data])
    df = df[df.room == room]

    no_goals = df.goals.apply(len)
    print("Multi goal", no_goals.sum())
    df = df[no_goals > 1]
    df = df[df.class_name == "bottle"]

    for idx, row in df.iterrows():
        ep.start_position = row["start_position"]
        ep.start_rotation = row["start_rotation"]
        # ep.goals = row["goals"]
        ep.goals = [row["goals"][0]]
        ep.class_name = row["class_name"]
        ep.info = row["info"]
        yield ep


class AIMASRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        current_target_distance = self._distance_target()
        return current_target_distance

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_agent_state(self):
        return self._env.sim.get_agent_state()

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position

        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


apartment = "office_3"
ep = NavigationEpisode(
    episode_id="0",
    scene_id=f"/raid/workspace/alexandrug/Replica-Dataset/dataset"
             f"/{apartment}/habitat/mesh_semantic.ply",
    start_position=[0.1244944, -1.020689, -2.1761066],
    start_rotation=[0, 0.163276, 0, 0.98658],
    goals=[
        NavigationGoal(position=[-2.452475, -1.02069 ,  1.156327])
    ],
    info={"geodesic_distance": 0.001},
)
dataset = Dataset()
dataset.episodes = [ep]


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def example():
    cfg = habitat.get_config("configs/tasks/pointnav.yaml")
    cfg.defrost()
    cfg.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    cfg.TASK.SENSORS.append("HEADING_SENSOR")
    cfg.SIMULATOR.SCENE = f"/raid/workspace/alexandrug/Replica-Dataset/dataset/{apartment}/habitat/mesh_semantic.ply"
    cfg.freeze()
    env = AIMASRLEnv(config=cfg, dataset=dataset)

    global ep
    ep_iterator = get_goal(apartment, ep)

    # env = habitat.Env(
    #     config=habitat.get_config("configs/tasks/pointnav.yaml")
    # )
    #
    print("Environment creation successful")
    observations = env.reset()
    print(observations.keys())
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0], observations[
            "pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    print("GOAL:", env.current_episode.goals)
    print(habitat.SimulatorActions)

    count_steps = 0
    done = False
    while not done:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = habitat.SimulatorActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = habitat.SimulatorActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = habitat.SimulatorActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = habitat.SimulatorActions.STOP
            print("action: FINISH")
        elif keystroke == ord(LOOK_UP):
            action = habitat.SimulatorActions.LOOK_UP
            print("action: LOOK_UP")
        elif keystroke == ord(LOOK_DOWN):
            action = habitat.SimulatorActions.LOOK_DOWN
            print("action: LOOK_DOWN")
        elif keystroke == ord("r"):
            ep = next(ep_iterator)
            print(f"New_ep: {ep.class_name} @ {ep.goals}")
            env.episode_iterator = iter([ep])
            env.reset()
            continue
        elif keystroke == ord('q'):
            x = float(input("X:"))
            y = float(input("Y:"))
            z = float(input("Z:"))
            print(x, y, z)
            action = {
                "action": "TELEPORT",
                "action_args": {
                    "position": [x, y, z],
                    "rotation": [0, -0.39109465, 0, 0.92035]
                }
            }
            print("TELEPORT")
        else:
            print("INVALID KEY")
            continue

        observations, reward, done, info = env.step(action)
        count_steps += 1

        agent_state = env.get_agent_state()
        print(agent_state.position)
        print(reward)

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0], observations["pointgoal_with_gps_compass"][1]))

        im = observations["rgb"]
        top_down_map = draw_top_down_map(
            info, observations["heading"], im.shape[0]
        )
        output_im = np.concatenate((transform_rgb_bgr(im), top_down_map), axis=1)

        cv2.imshow("RGB", output_im)

    print("Episode finished after {} steps.".format(count_steps))

    if action == habitat.SimulatorActions.STOP and observations["pointgoal_with_gps_compass"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()
