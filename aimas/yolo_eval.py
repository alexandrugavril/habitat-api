import numpy as np
from yolov3.detector import YoloDetector
import torch
import os
import habitat
from habitat.tasks.nav.nav_task_multi_goal import CLASSES
import pprint
import yaml
from yolov3.models import Darknet
from yolov3.utils import utils
import cv2
import pandas as pd
import copy
import os

classes_list = np.array(list(CLASSES.values()))

path = "/raid/workspace/alexandrug/habitat-api/results/pointgoal_obj_inview/video_dir/episode_data"
path_save = "/raid/workspace/alexandrug/habitat-api/results/pointgoal_obj_inview/video_dir/episode_data_ann"

yolo_cfg = "/raid/workspace/alexandrug/habitat-api/yolov3/config/yolo_config.yaml"

files = os.listdir(path)
save_files = os.listdir(path_save)

cfg = habitat.get_config("configs/tasks/pointnav_replica2.yaml")


def init_model(cfg_path, device):

    opt = yaml.load(open(cfg_path))

    nms_thres = opt['nms_thres']
    # Set up model
    model = Darknet(opt['model_def'], img_size=opt['img_size']).to(device)

    if opt['weights_path'].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt['weights_path'])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt['weights_path']))

    model.eval()  # Set in evaluation mode
    classes = utils.load_classes(opt['class_path'])  # Extracts class labels from file
    return model, classes


def get_detections(x, y, w, h):
    xpx_low = ((x - w / 2) * w + w / 2)
    ypx_low = ((y - h / 2) * h + h / 2)

    xpx_high = ((x + w / 2) * w + w / 2)
    ypx_high = ((y + h / 2) * h + h / 2)

    result = torch.stack((xpx_low, ypx_low, xpx_high, ypx_high), 1)
    return result



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

OUT_FOLDER = "../Replica-Dataset/dataset_jsons/"
SAVE_KEYS = ["episode_id", "scene_id", "start_position", "start_rotation", "info", "goals",
             "t_coord", "t_size", "target_idx",  "room",
             "class_name",
             "complexity_quartile", "geodesic_distances", "euclidean_dist", "nav_point2goal"]


data = np.load("dataset_all_merged.npy", allow_pickle=True)


for xd in data:
    for g in xd.goals:
        g.position = list(g.position)
dfm = pd.DataFrame([x.__dict__ for x in data])

f = files[0]

full_path = os.path.join(path, f)
data = torch.load(full_path)



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model, coco_classes = init_model(yolo_cfg, device)
#model.eval()

#print(coco_classes)
#
# classes_dict = {}

#

#     data['annotated'] = np.zeros(len(data['rgb']))
#
#     images = data['rgb']
#     all_goals = data['goal_bbox_in_camera']
#     goal_class = data['goalclass']
#     indices = np.nonzero(goal_class)[:, 1].cpu()
#     selected_classes = np.take(classes_list, indices)
#
#     print(selected_classes)
#     # coco_indices = []
#     # for cls in selected_classes:
#     #     coco_indices.append(coco_classes.index(cls))
#
#     base_detections = get_detections(all_goals[:, 3], all_goals[:, 4],
#                                      all_goals[:, 5], all_goals[:, 6])
#     for idx, img in enumerate(images):
#         c_bbox = all_goals[idx]
#         c_class = selected_classes[idx]
#
#         dx = c_bbox[3]
#         dy = c_bbox[4]
#
#         wg = c_bbox[5]
#         hg = c_bbox[6]
#
#         w, h, _ = img.shape
#
#         xpx = int(dx * w + w/2)
#         ypx = int(dy * h + h/2)
#
#         xpx_low = int((dx - wg / 2) * w + w / 2)
#         ypx_low = int((dy - hg / 2) * h + h / 2)
#
#         xpx_high = int((dx + wg / 2) * w + w / 2)
#         ypx_high = int((dy + hg / 2) * h + h / 2)
#
#         im = img.cpu().numpy().astype('uint8').copy()
#         im = cv2.rectangle(im, (xpx_low, ypx_low), (xpx_high, ypx_high), (0, 255, 0), 1)
#         print(c_class)
#         key = 0
#         while key == 0:
#             cv2.imshow("Frame", im)
#             new_key = cv2.waitKey(0)
#             if new_key == ord('q'):
#                 print("OK")
#                 data['annotated'][idx] = 1
#                 key = new_key
#             elif new_key == ord('m'):
#                 data['annotated'][idx] = -1
#                 print("NOK")
#                 key = new_key
#
#     torch.save(data, os.path.join(path_save, f)
# )
#
#     # for coco in coco_indices:
#     #     if coco_classes[coco] in classes_dict:
#     #         classes_dict[coco_classes[coco]] += 1
#     #     else:
#     #         classes_dict[coco_classes[coco]] = 1
#     # continue
#
# print(classes_dict)
# print(sum(classes_dict.values()))
# exit()

    # images = images.transpose(1, 3).to(device)
    # with torch.no_grad():
    #     detections = model(images)
    #     for i in range(1, 100, 2):
    #         conf_thres = i / 100.0
    #         final_detections = utils.non_max_suppression(detections, conf_thres, 0.4)
    #         for dets, idx in enumerate(final_detections):
    #             cls_det = dets.cpu().numpy()
    #             cls_det = cls_det[cls_det[:, -1] == coco_indices[idx]]

