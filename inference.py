
# coding: utf-8

# In[ ]:

import tensorflow as tf
import math
import cv2
import os.path as osp
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper

from model import dbnet
from model_resnet_custom import DBNet_res50, DBNet_res18

import time

print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 保证长边=640，长宽比近似不变
def resize_image(image, image_long_side=640):  # 736
    height, width, _ = image.shape
    if height < width:
        scale = image_long_side / width
        new_width = image_long_side
        new_height = int(scale * height)
        new_height = int(math.ceil(new_height / 32) * 32)
    else:
        scale = image_long_side / height
        new_height = image_long_side
        new_width = int(scale * width)
        new_width = int(math.ceil(new_width / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    # 计算 box 包围的区域的平均得分
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    contour = contour.astype(np.float32)
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    pred = pred[..., 0]
    # print('pred[..., 0].shape: ', pred.shape)
    bitmap = bitmap[..., 0]
    # print('bitmap[..., 0].shape: ', bitmap.shape)
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


if __name__ == '__main__':
#     mean = np.array([103.939, 116.779, 123.68])

#    model_weights = './models/tf_SROIE2019_resnet18_db_weights.h5'  
    model_weights = './checkpoints/2020-02-13/tf_SROIE2019_resnet18_db_95_1.2897_1.3219_weights.h5'
#     _, model = dbnet()
    _, model = DBNet_res18()
    
#     model.load_weights(model_weights, by_name=True, skip_mismatch=True)
    model.load_weights(model_weights, by_name=True)
    
    # model.save_weights('./models/db_48_2.0216_2.5701_predict_weights.h5')
    # model.save('./models/db_48_2.0216_2.5701_predict_model.h5', include_optimizer=True)

    start = time.time()
    n = 0
    for image_path in glob.glob(osp.join('./test_images', '*.jpg')):
        # print('img_path: ', image_path)
        image = cv2.imread(image_path)
        src_image = image.copy()
        h, w = image.shape[:2]
        image = resize_image(image)
        image = image.astype(np.float32)
#         image -= mean
        image_input = np.expand_dims(image, axis=0)
        p = model.predict(image_input)[0]
        bitmap = p > 0.3  # 0.3
        # print('len_bitmap: ', len(bitmap))
        # print('bitmap.shape: ', bitmap.shape)
        boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.5)  # 0.5
        for box in boxes:
            # 画矩形框
#             x_min = min(np.array(box)[:, 0])
#             x_max = max(np.array(box)[:, 0])
#             y_min = min(np.array(box)[:, 1])
#             y_max = max(np.array(box)[:, 1])
#             cv2.rectangle(src_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 画不规则框
            cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)

        image_fname = osp.split(image_path)[-1]
        cv2.imwrite('test_results/' + image_fname, src_image)
        n = n + 1
    stop = time.time()
    print('Spend time per img: ', (stop-start) / n)

