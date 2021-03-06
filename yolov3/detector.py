import torch
import cv2
import yaml
from yolov3.models import *
from yolov3.utils.datasets import *
from yolov3.utils import utils
from yolov3 import models


class YoloDetector:
    def __init__(self, cfg_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opt = yaml.load(open(cfg_path))
        self.conf_thres = self.opt['conf_thres']
        self.nms_thres = self.opt['nms_thres']
        # Set up model
        self.model = Darknet(self.opt['model_def'], img_size=self.opt['img_size']).to(device)
        self.img_size = self.opt['img_size']

        if self.opt['weights_path'].endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.opt['weights_path'])
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.opt['weights_path']))

        self.model.eval()  # Set in evaluation mode
        self.classes = utils.load_classes(self.opt['class_path'])  # Extracts class labels from file

        out_size = 32
        mode = "nearest"
        self.b1_scale = nn.Upsample(scale_factor=out_size // 8, mode=mode)
        self.b2_scale = nn.Upsample(scale_factor=out_size // 16, mode=mode)
        self.b3_scale = nn.Upsample(scale_factor=out_size // 32, mode=mode)

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x

        print(pad_x)
        print(pad_y)

        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes

    def view_heatmap(self, detections):
        bs = detections.size(0)

        b1, b2, b3 = (detections[:, :192], detections[:, 192: 960],
                      detections[:, 960:])

        ordd = (0, 1, 4, 2, 3)

        b1 = b1.view(bs, 3, 8, 8, 85).permute(*ordd).contiguous().view(bs, -1, 8, 8)
        b2 = b2.view(bs, 3, 16, 16, 85).permute(*ordd).contiguous().view(bs, -1, 16, 16)
        b3 = b3.view(bs, 3, 32, 32, 85).permute(*ordd).contiguous().view(bs, -1, 32, 32)

        b1 = self.b1_scale(b1)
        b2 = self.b2_scale(b2)
        b3 = self.b3_scale(b3)

        out = (b1 + b2 + b3) / 3
        out = out.view(bs, 3, 85, 32, 32)
        out = out.mean(dim=1)

        class_prob = out[0, 5 + 56].detach().cpu().numpy()
        class_prob = class_prob / out[0, 5:].max().item()

        view = cv2.resize(class_prob, (256, 256))
        cv2.imshow("Heat", view)
        print(out.size())

    def detect(self, img, conf_thres, display = False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t_img = torch.from_numpy(img.astype('float') / 255.0).cuda().permute(2, 0, 1).unsqueeze(0)
        t_img = t_img.type(torch.cuda.FloatTensor)
        detections = self.model(t_img)

        print(self.view_heatmap(detections))

        print(detections.size())
        detections = utils.non_max_suppression(detections, conf_thres, self.nms_thres)[0]
        # Draw bounding boxes and labels of detections
        img_disp = img.copy()

        if detections is not None:
            # Rescale boxes to original image
            detections = self.rescale_boxes(detections, self.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                print("{} {} {} {}".format(x1, y1, x2, y2))
                # Create a Rectangle patch
                img_disp = cv2.rectangle(img_disp, (x2, y2), (x1, y1), (255, 0, 0), 2)

        if display:
            cv2.imshow("Test", img_disp)
            cv2.waitKey(0)

        return detections, img_disp

import torch

import numpy as np
import cv2

# draw cube
w = 256
hw = w //2
img = np.zeros((w, w), dtype=np.uint8)
img.fill(10)

cube_front_w = 100 // 2
cube_back_w = 40 // 2
cube_depth_w = 20
cube_start_depth = 1
cube_back_depth = 7


img[hw-cube_front_w: hw+cube_front_w, hw-cube_front_w: hw+cube_front_w] = cube_start_depth

depths = np.linspace(cube_back_depth, cube_start_depth, cube_depth_w)
depth_ws = np.linspace(cube_back_w, cube_front_w, cube_depth_w)
for i, row in enumerate(range(hw-cube_front_w-cube_depth_w, hw-cube_front_w)):
    offset = int(depth_ws[i])
    img[row, hw-offset: hw+offset] = depths[i]

if __name__ == "__main__":
    import glob
    import numpy as np
    import torch

    yolo = YoloDetector('/raid/workspace/alexandrug/habitat-api/yolov3/config/yolo_config.yaml')

    files = glob.glob("/raid/workspace/alexandrug/habitat-api/results"
                   "/pointgoal_obj_inview/video_dir/episode_data/*")

    for f in files:
        data = torch.load(f)
        for i in range(len(data["rgb"])):
            img = data["rgb"][i].cpu().numpy().astype(np.uint8)

            img = cv2.resize(img, (256, 256))
            print(img.shape)
            yolo.detect(img, 0.6, True)
