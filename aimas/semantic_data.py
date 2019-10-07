import pandas as pd
import glob
import os
import json
import collections
import numpy as np
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
import quaternion

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 100)

DATA_PATH = "/raid/workspace/alexandrug/Replica-Dataset/dataset"
OUT_DATA_PATH = DATA_PATH + "/semantic_data.npy"

MATCHING_CLASSES = dict({  # REPLICA class name: COCO class name
    "book": "book",
    "chair": "chair",
    "table": "diningtable",
    "bowl": "bowl",
    "bottle": "bottle",
    "indoor-plant": "pottedplant",
    "cup": "cup",
    "vase": "vase",
    "tv-screen": "tvmonitor",
    "sofa":"sofa",
    "bike": "bicycle",
    "sink": "sink"
})


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_data():
    semantic_files = glob.glob(DATA_PATH + "/**/semantic.json", recursive=True)

    df_semantics = []
    df_objectss = []

    for smf in semantic_files:
        room_name = os.path.basename(os.path.dirname(smf))

        semantic = json.load(open(smf, "r"))  # type: dict
        semantic.pop("segmentation")
        df_semantic = pd.DataFrame(flatten(semantic), index=[0])
        df_semantic["room"] = room_name

        info_semantic = json.load(open(os.path.dirname(smf) +
                                       "/habitat/info_semantic.json", "r"))
        for rmk in ["classes", "id_to_label"]:
            info_semantic.pop(rmk)

        df_objects = pd.DataFrame([flatten(x)
                                   for x in info_semantic["objects"]])
        df_objects["room"] = room_name

        gravity_center = np.array(list(semantic["gravityCenter"].values()))
        gravity_dir = np.array(list(semantic["gravityDirection"].values()))

        centers = np.stack(df_objects.oriented_bbox_abb_center.values)

        df_semantics.append(df_semantic)
        df_objectss.append(df_objects)

    df_semantic = pd.concat(df_semantics)
    df_objects = pd.concat(df_objectss)

    np.save(OUT_DATA_PATH,
            {"df_semantic": df_semantic, "df_objects": df_objects})


def convert_replica_coord(coord, rotation):
    rotation_replica_habitat = quat_from_two_vectors(np.array([0, 0, -1]),
                                                     np.array([0, -1, 0]))

    obj_rotation = quaternion.from_float_array([rotation[-1]] + rotation[:-1])

    obj_coord = quat_rotate_vector(rotation_replica_habitat * obj_rotation,
                                   coord)
    return obj_coord


def analyze_data():
    semantic_data = np.load(OUT_DATA_PATH, allow_pickle=True).item()

    df_semantic = semantic_data["df_semantic"]
    df_objects = semantic_data["df_objects"]

    df_objects["habitat_coord"] = \
        list(map(convert_replica_coord,
                 df_objects.oriented_bbox_abb_center.values,
                 df_objects.oriented_bbox_orientation_rotation.values))

    df_objects["habitat_size"] = \
        list(map(convert_replica_coord,
                 df_objects.oriented_bbox_abb_sizes.values,
                 df_objects.oriented_bbox_orientation_rotation.values))

    np.save(OUT_DATA_PATH,
            {"df_semantic": df_semantic, "df_objects": df_objects})

    for idx, row in df_objects.sample(10).iterrows():
        print(row["class_name"], row["habitat_coord"], row["habitat_size"])
