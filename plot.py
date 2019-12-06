import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
import pandas as pd
import seaborn as sns; sns.set()
import numpy as np


def get_name(path):
    name = re.findall("tb/(.*)/events", path)[0]
    return name


prefix = "relstart_"

folders = glob.glob(f"results/*{prefix}*")
experiments = [
    (re.findall(prefix + "(.*)", x)[0], x) for x in folders
]


# experiments = [
#     ("batch2", "results/2019-11-17_09-21-31_batch2/"),
#     ("batch3", "results/2019-11-17_09-25-35_batch3/"),
#     ("noise_action_1", "results/2019-11-17_19-06-11_batch2_noise_action_1/"),
#     ("noise_rgb", "results/2019-11-17_19-07-06_batch2_noise_rgb/"),
#     ("noop", "results/2019-11-17_19-08-34_batch2_noop/"),
#     ("noise_depth", "results/2019-11-17_19-10-03_batch2_noise_depth/"),
#     ("noise_tilt", "results/2019-11-17_19-16-33_batch2_noise_tilt/"),
#     ("cnnrelu", "results/2019-11-18_12-08-18_batch2_cnnrelu/"),
#     ("cnnrelu_drop", "results/2019-11-18_12-09-45_batch2_cnnrelu_drop/"),
# ]


# experiments = [
#     ("Explorare_catre_obiect",
#      "/raid/workspace/alexandrug/habitat-api/results/single_point_goal_inview/tb/"),
# ]


data = dict()
losses_names = set()

for exp_name, exp_path in experiments:
    loss_files = glob.glob(f"{exp_path}/explore_GO_test/tb/losses_*/events.out"
                           f".tfevents*")

    print(f"Start reading {exp_path}")
    data[exp_name] = dict()
    for event_file in loss_files:
        loss_name = get_name(event_file)
        losses_names.add(loss_name)

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        data[exp_name][loss_name] = dict({"value": [], "step": []})
        for value in ea.Scalars("losses"):
            data[exp_name][loss_name]["value"].append(value.value)
            data[exp_name][loss_name]["step"].append(value.step)

    print(f"Finished reading {exp_name}")

print(losses_names)
rolling_len = 50
for loss_name in losses_names:
    fig, ax = plt.subplots()

    for exp_name in data.keys():
        print(exp_name)
        loss_data = data[exp_name][loss_name]
        steps = loss_data["step"]
        value = pd.Series(loss_data["value"])

        win = value.rolling(rolling_len)
        mu = win.mean()
        sigma = win.std()

        base_line, = ax.plot(steps, mu, label=exp_name)
        ax.fill_between(steps, mu + sigma, mu - sigma,
                        facecolor=base_line.get_color(), alpha=0.5)

    ax.legend(loc="upper right")
    ax.set_title(loss_name)
    plt.show()
