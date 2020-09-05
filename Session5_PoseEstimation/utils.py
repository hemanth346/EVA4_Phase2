import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import pyautogui as pg
import keyboard 

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

keypoints = ['r-ankle', 'r-knee', 'r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis', 'thorax', 'upper-neck', 'head-top', 'r-wrist', 'r-elbow', 'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']

skeleton_parts = {
    'rleg' : (1,0), 'rthigh' : (2,1),
    'lleg' : (4,5), 'lthigh' : (4,3),
    'rhip' : (2,6), 'lhip' : (3,6),
    'torso' : (7,6), 'neck' : (7,8), 'head' : (8,9), 
    'lcollar' : (7,13), 'rcollar' : (7,12),
    'lbiceps' : (13, 14), 'lhand' : (14, 15), 
    'rbiceps' : (12, 11), 'rhand' : (11, 10)
}

def get_position(pointA, pointB, threshold=15):
        # points in format (height, width)
        # y, x = (height, width) 
        # threshold -> Acceptable distance to consider as near
        # if  using third refence point, we can get both up and down wrt to it, 
        # here down implies not up and viceversa
        Ay, Ax = pointA
        By, Bx = pointB
        is_near = (abs(Ax-Bx) < threshold and abs(Ay-By) < threshold)
        if is_near:
            return 'near'

        if (abs(Ax-Bx) < threshold or abs(Ay-By) < threshold):
            if abs(Ax-Bx) < threshold:
                return 'same_width'
            return 'same_height'

        if (Ax-Bx) > 0: 
            # if positive => towards right; less than given threshold will be near already
            position = 'right_'
        else:
            position = 'left_'

        if (Ay-By) > 0: 
            # if positive => high
            position += 'up'
        else:
            position += 'down'
        return position

def control_mouse(position):
    commands = {
        'near': pg.dragRel(5, 5), #pg.click(button='left', clicks=1), 
        'same_width': pg.dragRel(30, 0), #pg.click(button='right', clicks=2, interval=0.25),
        'same_height': pg.dragRel(0, 30),#pg.scroll(30),
        'right_up': pg.dragRel(30, 30),
        'right_down': pg.dragRel(30, -30),
        'left_up': pg.dragRel(-30, 30),
        'left_down': pg.dragRel(-30, -30)
        }
    print(pg.position())
    commands[position]
    return True

def control_keyboard(position):
    if 'right' in position:
        keyboard.press_and_release('right')
        print('right clicked')
    if 'left' in position:
        keyboard.press_and_release('left')
        print('left clicked')
    return

# def show_pose_maps(image_path, model=model):
#     pose_maps = get_heatmaps(image_path, model)
#     pred_coordintates, pred_val = get_point_loc_conf(pose_maps)
#     pred_confidence = sigmoid_v(pred_val)
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image,pose_maps.shape[1:])
#     plt.figure(figsize=(18, 18))
#     for idx, pose in enumerate(pose_maps):
#         plt.subplot(4, 4, idx + 1)
#         plt.title(f'{keypoints[idx]}, Confidence : {round(pred_confidence[idx],4)}')
#         plt.imshow(image, cmap='gray', interpolation='bicubic')
#         plt.imshow(pose, alpha=0.5, cmap='jet', interpolation='bicubic')
#         plt.axis('off')
#     plt.show() 
