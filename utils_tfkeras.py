
# coding: utf-8

# In[ ]:

from tensorflow import keras
# import keras

import numpy as np
import cv2
import os 
import math
import random
import time

import pyclipper
from shapely.geometry import Polygon
import imgaug
from imgaug import augmenters as iaa

# In[ ]:


epsilon = 1e-9
aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0,2.0)),  # 高斯模糊增强器
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # 高斯噪声增强器
    iaa.AddToHueAndSaturation((-50, 50))  # color jitter, only affects the image
])

# 旋转
def rotate_img_bbox(img, anns, rot_angle=10, scale=1.):
    #---------------------- 旋转图像 ----------------------
    new_anns = []
    angle = np.random.randint(-rot_angle, rot_angle)
    bboxes = []
    for ann in anns:
        bboxes.append([ann['poly'][0], ann['poly'][1],
                       ann['poly'][2], ann['poly'][3]])
    
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4, borderValue=(255,255,255))

    #---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    new_anns = []
    for i, bbox in enumerate(bboxes):
        point1 = np.dot(rot_mat, np.array([bbox[0][0], bbox[0][1], 1]))
        point2 = np.dot(rot_mat, np.array([bbox[1][0], bbox[1][1], 1]))
        point3 = np.dot(rot_mat, np.array([bbox[2][0], bbox[2][1], 1]))
        point4 = np.dot(rot_mat, np.array([bbox[3][0], bbox[3][1], 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        # 加入list中
        new_ann = {'poly': concat.tolist(), 'text': anns[i]['text']}
        new_anns.append(new_ann)
        
    return rot_img, new_anns

# In[ ]:

def resize(size, image, anns):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
#     padimg = np.zeros((size, size, c), image.dtype)
    padimg = np.full(shape=(size,size,c), fill_value=255, dtype=float)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.float64)
        poly *= scale
        new_ann = {'poly': poly.tolist(), 'text': ann['text']}
        new_anns.append(new_ann)
    return padimg, new_anns

# 重整顶点：顺时针调整       
def reorder_vertexes(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype = "float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def load_all_anns(gt_paths):
    res = []
    for gt in gt_paths:
        lines = []
        with open(gt, mode='r', encoding='UTF-8') as fr:
            reader = fr.readlines()
            for line in reader:
                item = {}
                line = line.strip().split(',')
                label = line[-1]
                try:
                    i = [int(ii) for ii in line[:8]]
                    poly = np.array(i).reshape((-1, 2))
                    poly = reorder_vertexes(poly)
                except:
                    print('wrong poly')
                    continue
                item['poly'] = poly.astype(np.int32).tolist()
                item['text'] = label
                lines.append(item)
        res.append(lines)
#     print('num imgs: ', len(res))
    return res

def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # padded_polygon ： 文字区域padded后的区域，相反与 shrinked 操作
    padded_polygon = np.array(padding.Execute(distance)[0])
    # mask ： 文字区域padded后的区域的标识图
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
# + epsilon
    cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2) + epsilon)
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
# + epsilon
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance + epsilon)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result

class DataGenerator(keras.utils.Sequence):
    # list_IDs: 图片名list   labels： 图片名对应的label_file list
    def __init__(self,list_IDs, labels, batch_size=16, image_size=640, min_text_size=8, shrink_ratio=0.4, thresh_min=0.3,
             thresh_max=0.7, is_training=True, shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.is_training = is_training
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        
        x, y = self.__data_generation(list_IDs_temp, labels_temp)
        return x, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, label_IDs_temp):
        # list_IDs: 图片名list   labels： 图片名对应的label_file list        
        batch_images = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.float32)
        batch_gts = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_thresh_maps = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        batch_thresh_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
        
        mean = [0.0, 0.0, 0.0]

#         all_anns:
#         [{'poly': [[72, 25], [326, 25], [326, 64], [72, 64]], 'text': 'TAN'}, {'poly': [[50, 82], [440, 82], [440, 121], [50, 121]],'text': 'BOOK'},
#         {'poly': [[205, 121], [285, 121], [285, 139], [205, 139]], 'text': '789417-W'}, {}, ...]
        all_anns = load_all_anns(label_IDs_temp)
        for ii, path in enumerate(list_IDs_temp): 
            
            image = cv2.imread(path)
            anns = all_anns[ii]

            if self.is_training:
                image, anns = rotate_img_bbox(image, anns, rot_angle=20, scale=1.)

            image, anns = resize(self.image_size, image, anns)
            # show_polys(image.copy(), anns, 'after_aug')
            # cv2.waitKey(0)
            anns = [ann for ann in anns if Polygon(ann['poly']).is_valid]
            # gt 图片中文字区域经过shrinking（收缩）后的标识图
            gt = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            # mask 图片中有效区域的标识图（能够用来计算正负样本loss的区域）
            mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
            thresh_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            thresh_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            for ann in anns:
                poly = np.array(ann['poly'])
                height = max(poly[:, 1]) - min(poly[:, 1])
                width = max(poly[:, 0]) - min(poly[:, 0])
                polygon = Polygon(poly)
                # generate gt and mask
                if polygon.area < 1 or min(height, width) < self.min_text_size or ann['text'] == '###':
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    # distance： shrinking的距离
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(l) for l in ann['poly']]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    # shrinked: shrinked 收缩后的区域, 由多个点所表示
                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 0:
                        cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
                    else:
                        shrinked = np.array(shrinked[0]).reshape(-1, 2)
                        if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                            cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                        else:
                            cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                            continue
                # generate thresh map and thresh mask
                # thresh_mask : 文字区域padded后的区域的标识图
                draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=self.shrink_ratio)
            thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

#             image = image.astype(np.float32)
            image[..., 0] -= mean[0]
            image[..., 1] -= mean[1]
            image[..., 2] -= mean[2]
            
            
            batch_images[ii] = image
            batch_gts[ii] = gt  
            batch_masks[ii] = mask
            batch_thresh_maps[ii] = thresh_map
            batch_thresh_masks[ii] = thresh_mask
            
        if self.is_training:
            batch_images = batch_images.astype(np.uint8)
            batch_images = aug.augment_images(batch_images)
            
        batch_gts = np.expand_dims(batch_gts, axis=-1)
        batch_masks = np.expand_dims(batch_masks, axis=-1)
        batch_thresh_maps = np.expand_dims(batch_thresh_maps, axis=-1)
        batch_thresh_masks = np.expand_dims(batch_thresh_masks, axis=-1)
            
        batch_outputs = np.concatenate((batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks), axis=-1)
#             batch_outputs[ii] = outputs
        
        # return batch_outputs, batch_loss
        return batch_images, batch_outputs
