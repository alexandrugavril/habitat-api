import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 300)

DATA_PATH = "dataset_all.npy"

data = np.load(DATA_PATH, allow_pickle=True)


df = pd.DataFrame([x.__dict__ for x in data])

no_goals = df.goals.apply(len)

# -- Fix
multi_goal = df[no_goals > 1]
for idx, row in multi_goal.iterrows():
    row["goals"][0].position = row["t_coord"]



df["tested_coord"] = df[["goals", "target_idx"]].\
    apply(lambda r: r["goals"][r["target_idx"]].position, axis=1)


dist_goal_to_test = (df.t_coord - df.tested_coord).apply(np.linalg.norm)

dist_to_test = np.linalg.norm(np.stack(df.t_coord.values)[:, [0, 1]] -
                              np.stack(df.start_position.values)[:, [0, 1]], axis=1)


geodesic_ratio = df["info"].apply(lambda x: x["geodesic_distance"]) / dist_to_test
geodesic_ratio[(df.goals.apply(len) == 1)]


plt.hist(df["info"].apply(lambda x: x["geodesic_distance"]).values, bins=100)
plt.show()
