import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import habitat
import numpy as np


pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 300)


def analytics():
    DATA_PATH = "dataset_all.npy"

    data = np.load(DATA_PATH, allow_pickle=True)

    df = pd.DataFrame([x.__dict__ for x in data])

    no_goals = df.goals.apply(len)

    df["tested_coord"] = df[["goals", "target_idx"]].\
        apply(lambda r: r["goals"][r["target_idx"]].position, axis=1)

    dist_goal_to_test = (df.t_coord - df.tested_coord).apply(np.linalg.norm)

    dist_to_test = np.linalg.norm(np.stack(df.t_coord.values)[:, [0, 1]] -
                                  np.stack(df.start_position.values)[:, [0, 1]],
                                  axis=1)


    geodesic_ratio = df["info"].apply(lambda x: x["geodesic_distance"]) / dist_to_test
    geodesic_ratio[(df.goals.apply(len) == 1)]


    plt.hist(df["info"].apply(lambda x: x["geodesic_distance"]).values, bins=100)
    plt.title("Target goal geodesic distance")
    plt.show()

    plt.hist(geodesic_ratio[(df.goals.apply(len) == 1)], bins=100, log=True)
    plt.title("Single Goal Ratio")
    plt.show()

    plt.hist(geodesic_ratio[(df.goals.apply(len) > 1)], bins=100, log=True)
    plt.title("Multi goal Ration")
    plt.show()

from habitat.datasets.utils import get_action_shortest_path

# -- Merge goals
def merge_goals():

    env = habitat.Env(
        config=habitat.get_config("configs/tasks/pointnav.yaml")
    )
    sim = env.sim

    observations = env.reset()

    DATA_PATH = "dataset_all.npy"

    data = np.load(DATA_PATH, allow_pickle=True)

    df = pd.DataFrame([x.__dict__ for x in data])
    df.t_coord = df.t_coord.apply(tuple)
    merged_ep = []

    for room, room_group in df.groupby(["room"]):
        print(f"ROOM: {room}")
        for class_name, class_group in room_group.groupby(["class_name"]):

            env.episode_iterator = iter([data[class_group.index[0]]])
            env.reset()

            target_group = class_group.groupby(['t_coord']).head(1)

            all_goals = target_group.goals.values
            goals = reduce(lambda x, y: x + y, all_goals)
            target_idx = [[ix] * len(all_goals[ix]) for ix in range(len(
                all_goals))]
            target_idx = reduce(lambda x, y: x + y, target_idx)

            t_coord = [x for x in target_group.t_coord.values]
            t_size = [x for x in target_group.t_size.values]
            tested_coord = [x for x in target_group.tested_coord.values]

            for idx in class_group.index:
                data[idx].goals = goals
                data[idx].t_coord = t_coord
                data[idx].t_size = t_size
                data[idx].tested_coord = tested_coord
                data[idx].target_idx = target_idx

                #\frac{geodesicDist2D + navigablePointDistToObj2D}{euclDist2D}
                #  * \frac{1}{(wObj * hObj * lObj)^{\frac{1}{3}}}

                start_position = np.array(data[idx].start_position)
                t_coord = np.array(data[idx].t_coord)

                t_positions = np.array([t.position for t in goals])
                t_coord_coresponding = t_coord[target_idx]

                geodesic_distances = np.array([
                    sim.geodesic_distance(start_position, t) for t in
                    t_positions])
                #
                # # -- calculate paths
                # paths = []
                # for t in t_positions:
                #     angles = [x for x in range(-180, 180, 30)]
                #     angle = np.radians(np.random.choice(angles))
                #     source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
                #     p = get_action_shortest_path(
                #         sim,
                #         source_position=start_position,
                #         source_rotation=source_rotation,
                #         goal_position=t,
                #         success_distance=0.2,
                #         max_episode_steps=500,
                #         shortest_path_mode="geodesic_path",
                #     )
                #     paths.append(p)

                nav_point2goal = np.linalg.norm(t_positions[:, [0, 2]] -
                                                t_coord_coresponding[:, [0, 2]]
                                                , axis=1)

                euclidean_dist = np.linalg.norm(start_position[[0, 2]] -
                                                t_coord[:, [0, 2]], axis=1)
                euclidean_dist = euclidean_dist[target_idx]

                size_m = np.power(np.abs(np.array(t_size).prod(axis=1)), 1/3.)
                size_m = size_m[target_idx]

                # Formula for complexity score
                complexity_score = (geodesic_distances + nav_point2goal) / (
                    euclidean_dist)
                complexity_score = np.clip(complexity_score, 1, np.inf) * (1 / size_m)

                data[idx].geodesic_distances = geodesic_distances
                data[idx].euclidean_dist = euclidean_dist
                data[idx].nav_point2goal = nav_point2goal
                data[idx].complexity_score = complexity_score

    np.save("dataset_all_merged", data)


# =============================================================================
# -- Export train / val / test - splits

import copy
import os

APARTMENTS = [
    'office_1', 'office_2', 'room_2', 'frl_apartment_0', 'office_3',
    'frl_apartment_2', 'hotel_0', 'apartment_0', 'frl_apartment_5',
    'room_1', 'room_0', 'apartment_2', 'apartment_1',
    'frl_apartment_4', 'office_4', 'office_0', 'frl_apartment_1',
    'frl_apartment_3'
]

TEST_ROOMS = ["frl_apartment_0", "apartment_1", "hotel_0"]
VAL_ROOMS = ["frl_apartment_1", "office_2", ]
TRAIN_ROOMS = [x for x in APARTMENTS if x not in TEST_ROOMS and x not in
               VAL_ROOMS]

OUT_FOLDER = "../Replica-Dataset/"
SAVE_KEYS = ["episode_id", "scene_id", "start_position", "start_rotation", "info", "goals",
             "t_coord", "room",
             "complexity_quartile", "geodesic_distances", "euclidean_dist", "nav_point2goal"]


data = np.load("dataset_all_merged.npy", allow_pickle=True)

for xd in data:
    for g in xd.goals:
        g.position = list(g.position)
dfm = pd.DataFrame([x.__dict__ for x in data])

# -- Digitize complexity in quartiles
dfm["min_complexity_score"] = dfm.complexity_score.apply(min)
quartiles = dfm.min_complexity_score.describe()[["min", "25%", "50%", "75%", "max"]].values
quartiles[-1] += 1
dfm["complexity_quartile"] = np.digitize(dfm.min_complexity_score.values, quartiles)


# -- Save keys
dfm = dfm[SAVE_KEYS]

# -- Transform according to format
dfm.geodesic_distances = dfm.geodesic_distances.apply(list)
dfm.euclidean_dist = dfm.euclidean_dist.apply(list)
dfm.nav_point2goal = dfm.nav_point2goal.apply(list)
dfm.t_coord = dfm.t_coord.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
dfm["shortest_paths"] = None
dfm["start_room"] = None
dfm.scene_id = dfm.scene_id.apply(lambda x: x.replace(OUT_FOLDER, ""))
dfm.loc[:, "episode_id"] = np.arange(0, len(dfm))
#

model_ep = copy.deepcopy(data[0])


for dataset_name, datasate_rooms in [("train", TRAIN_ROOMS), ("val", VAL_ROOMS),
                                     ("test", TEST_ROOMS)]:
    dfmslice = dfm[dfm.room.isin(datasate_rooms)]
    dataset_elements = []
    for x in dfmslice.to_dict(orient="records"):
        p = copy.deepcopy(model_ep)
        p.__dict__ = x
        dataset_elements.append(x)

    dataset = habitat.Dataset()
    dataset.episodes = dataset_elements

    json_data = dataset.to_json()
    json_path = OUT_FOLDER + f"{dataset_name}.json"
    with open(json_path, "w") as f:
        f.write(json_data)

    os.system(f"gzip {json_path}")

    """ run """
if __name__ == "__main__":
    merge_goals()
