"""
Pose estimation with MPII pretrained weights
"""

import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch 
import torchvision.transforms as T
from utils import sigmoid_v, keypoints, skeleton_parts, get_position, control_mouse, control_keyboard

from queue import Queue
from threading import Thread
import time

sys.path.insert(0, os.path.abspath('human-pose-estimation.pytorch/lib'))
import models

# load the model
model = torch.load('mpii_model.pt')
model.eval() # set to evaluation


# results
output_coords = [] # considering aspect ratio
pred_confidence = [] # sigmoid of maxVal


def get_heatmaps(cv2_image, model=model):
    image = cv2_image # frame from video
    image = Image.fromarray(image)
    # image = Image.open(image_path).convert('RGB') # directly from path
    transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])
    image = transform(image)
    output = model(image.unsqueeze(0))
    output = output.squeeze(0)
    return output.cpu().detach().numpy()

def get_point_loc_conf(heatmaps):
    pred_coordintates = []
    pred_val = []

    for keypoint_channel in heatmaps:
        minVal, maxVal, minLoc, point = cv2.minMaxLoc(keypoint_channel)
        pred_coordintates.append(point)
        pred_val.append(maxVal)
    return pred_coordintates, pred_val


def get_pred(cv2_image, model=model, threshold=0.65):
    pred_maps = get_heatmaps(cv2_image, model=model)
    pred_coordintates, pred_val = get_point_loc_conf(pred_maps)
    pred_confidence = sigmoid_v(pred_val)

    ip_height, ip_width = cv2_image.shape[:-1] # h, w, c
    op_width, op_height = pred_maps.shape[1:] # batch, w, h
    aspect_ratio = [ip_width/op_width, ip_height/op_height]
    output_coords = [tuple(int(s1*s2) for s1, s2 in zip(aspect_ratio, co_ord)) for co_ord in pred_coordintates]

    # # drawing points for predictions
    # for coord in output_coords:
    #     cv2.circle(cv2_image, coord, 5, (0, 0, 255))

    # drawing lines
    for name, idxs in skeleton_parts.items():
        i, j = idxs
        if pred_confidence[i] < threshold or pred_confidence[j] < threshold:
            # print(f'{name} doesnot satisfy threshold of {threshold}.. Current values - {pred_confidence[i]} {pred_confidence[j]}')
            continue

        cv2.line(cv2_image, output_coords[i], output_coords[j], (0, 255, 0), 3)
    return cv2_image, dict(zip(keypoints, list(zip(output_coords, pred_confidence, pred_coordintates))))


# vid = cv2.VideoCapture('vid.mp4')
vid = cv2.VideoCapture(0)
print('Select application, waiting for 5 seconds..')
time.sleep(5)
count = 0
while (vid.isOpened()):
    check, frame = vid.read()
    count += 1 
    # cv2.imshow("image", frame) 
    if check is False:
        break
    pose, meta  = get_pred(frame, model, 0.60)
    rwrist, lwrist = meta['r-wrist'],meta['l-wrist']
    # print(rwrist, lwrist)

    # using prediction coordinates as the aspect ratio can vary -
    # a lot and might be difficult to get a generalized position
    position = get_position(rwrist[2], lwrist[2])
    print(position)
    # control_mouse(position)
    control_keyboard(position)
    cv2.imshow("image", pose) # show skeleton
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
