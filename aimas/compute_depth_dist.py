
import cv2
import os
import numpy as np
from itertools import groupby
from operator import itemgetter

img_dir = '/home/amiro/workspace/alex/habitat-api/images_pepper/'
entries = os.listdir(img_dir)
print(entries)
entries.sort()

vals = {}

for img_path in entries:
    im = cv2.imread(img_dir + img_path)

    rgb = im[:, 0:128, :]
    depth = im[:, 128:, :]

    crop_depth = depth[64-20:64+20, 64-20:64+20]
    crop_rgb = rgb[64-20:64+20, 64-20:64+20]
    avg_values = np.average(crop_depth[crop_depth.nonzero()])
    dict_key = img_path[0]
    if dict_key in vals:
        vals[dict_key].append(avg_values)
    else:
        vals[dict_key] = [avg_values]


for key in vals.keys():
    print((key, np.average(vals[key])))
