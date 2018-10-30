#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/10
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
##############################################################

# ����:
#     1. �ü�(��ı�bbox)
#     2. ƽ��(��ı�bbox)
#     3. �ı�����
#     4. ������
#     5. ��ת�Ƕ�(��Ҫ�ı�bbox)
#     6. ����(��Ҫ�ı�bbox)
#     7. cutout
# ע��:
#     random.seed(),��ͬ��seed,�������������һ����!!

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure


def show_pic(img, bboxes=None):
    '''
    ����:
        img:ͼ��array
        bboxes:ͼ�������boudning box list, ��ʽΪ[[x_min, y_min, x_max, y_max]....]
        names:ÿ��box��Ӧ������
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1��ʾԭͼ
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # ���ӻ���ͼƬ��С
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')


# ͼ���Ϊcv2��ȡ
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    # ������
    def _addNoise(self, img):
        '''
        ����:
            img:ͼ��array
        ���:
            ���������ͼ��array,�����������������[0,1]֮��,���Եó���255
        '''
        # random.seed(int(time.time()))
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True) * 255

    # ��������
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)  # flag>1Ϊ����,С��1Ϊ����
        return exposure.adjust_gamma(img, flag)

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        ԭ�汾��https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : �������
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxBΪ�����򣬷���iou
            boxBΪbouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # �õ�h��w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape

        mask = np.ones((h, w, c), np.float32)

        for n in range(n_holes):

            chongdie = True  # ���и�������Ƿ���box�ص�̫��

            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip����������������е�Ԫ��������a_min, a_max֮�䣬����a_max�ľ�ʹ�������� a_max��С��a_min,�ľ�ʹ��������a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # ��ת
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        �ο�:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        ����:
            img:ͼ��array,(h,w,c)
            bboxes:��ͼ�����������boundingboxs,һ��list,ÿ��Ԫ��Ϊ[x_min, y_min, x_max, y_max],Ҫȷ������ֵ
            angle:��ת�Ƕ�
            scale:Ĭ��1
        ���:
            rot_img:��ת���ͼ��array
            rot_bboxes:��ת���boundingbox����list
        '''
        # ---------------------- ��תͼ�� ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # �Ƕȱ仡��
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # ����任
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- ����bbox���� ----------------------
        # rot_mat�����յ���ת����
        # ��ȡԭʼbbox���ĸ��е㣬Ȼ�����ĸ���ת������ת�������ϵ��
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # �ϲ�np.array
            concat = np.vstack((point1, point2, point3, point4))
            # �ı�array����
            concat = concat.astype(np.int32)
            # �õ���ת�������
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # ����list��
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # �ü�
    def _crop_img_bboxes(self, img, bboxes):
        '''
        �ü����ͼƬҪ�������еĿ�
        ����:
            img:ͼ��array
            bboxes:��ͼ�����������boundingboxs,һ��list,ÿ��Ԫ��Ϊ[x_min, y_min, x_max, y_max],Ҫȷ������ֵ
        ���:
            crop_img:�ü����ͼ��array
            crop_bboxes:�ü����bounding box������list
        '''
        # ---------------------- �ü�ͼ�� ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # �ü���İ�������Ŀ������С�Ŀ�
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # ��������Ŀ������С����ߵľ���
        d_to_right = w - x_max  # ��������Ŀ������С���ұߵľ���
        d_to_top = y_min  # ��������Ŀ������С�򵽶��˵ľ���
        d_to_bottom = h - y_max  # ��������Ŀ������С�򵽵ײ��ľ���

        # �����չ�����С��
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # �����չ�����С�� , ��ֹ��õ�̫С
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # ȷ����ҪԽ��
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- �ü�boundingbox ----------------------
        # �ü����boundingbox�������
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # ƽ��
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        �ο�:https://blog.csdn.net/sty945/article/details/79387054
        ƽ�ƺ��ͼƬҪ�������еĿ�
        ����:
            img:ͼ��array
            bboxes:��ͼ�����������boundingboxs,һ��list,ÿ��Ԫ��Ϊ[x_min, y_min, x_max, y_max],Ҫȷ������ֵ
        ���:
            shift_img:ƽ�ƺ��ͼ��array
            shift_bboxes:ƽ�ƺ��bounding box������list
        '''
        # ---------------------- ƽ��ͼ�� ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # �ü���İ�������Ŀ������С�Ŀ�
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # ��������Ŀ����������ƶ�����
        d_to_right = w - x_max  # ��������Ŀ����������ƶ�����
        d_to_top = y_min  # ��������Ŀ����������ƶ�����
        d_to_bottom = h - y_max  # ��������Ŀ����������ƶ�����

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # xΪ��������ƶ�������ֵ,��Ϊ���Ҹ�Ϊ����; yΪ���ϻ��������ƶ�������ֵ,��Ϊ���¸�Ϊ����
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- ƽ��boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # ����
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            �ο�:https://blog.csdn.net/jningwei/article/details/78753607
            ƽ�ƺ��ͼƬҪ�������еĿ�
            ����:
                img:ͼ��array
                bboxes:��ͼ�����������boundingboxs,һ��list,ÿ��Ԫ��Ϊ[x_min, y_min, x_max, y_max],Ҫȷ������ֵ
            ���:
                flip_img:ƽ�ƺ��ͼ��array
                flip_bboxes:ƽ�ƺ��bounding box������list
        '''
        # ---------------------- ��תͼ�� ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:  # 0.5�ĸ���ˮƽ��ת��0.5�ĸ��ʴ�ֱ��ת
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # ˮƽ��ת
            flip_img = cv2.flip(flip_img, -1)
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- ����boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        '''
        ͼ����ǿ
        ����:
            img:ͼ��array
            bboxes:��ͼ������п�����
        ���:
            img:��ǿ���ͼ��
            bboxes:��ǿ��ͼƬ��Ӧ��box
        '''
        change_num = 0  # �ı�Ĵ���
        print('------')
        while change_num < 1:  # Ĭ��������һ��������ǿ��Ч
            if random.random() < self.crop_rate:  # �ü�
                print('�ü�')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)

            if random.random() > self.rotation_rate:  # ��ת
                print('��ת')
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.sample([90, 180, 270], 1)[0]
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if random.random() < self.shift_rate:  # ƽ��
                print('ƽ��')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if random.random() > self.change_light_rate:  # �ı�����
                print('����')
                change_num += 1
                img = self._changeLight(img)

            if random.random() < self.add_noise_rate:  # ������
                print('������')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  # cutout
                print('cutout')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                   threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:  # ��ת
                print('��ת')
                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)
            print('\n')
        # print('------')
        return img, bboxes


if __name__ == '__main__':

    ### test ###

    import shutil
    from xml_helper import *

    need_aug_num = 1

    dataAug = DataAugmentForObjectDetection()

    source_pic_root_path = './data_split'
    source_xml_root_path = './data_voc/VOC2007/Annotations'

    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            cnt = 0
            while cnt < need_aug_num:
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
                coords = parse_xml(xml_path)  # �����õ�box��Ϣ����ʽΪ[[x_min,y_min,x_max,y_max,name]]
                coords = [coord[:4] for coord in coords]

                img = cv2.imread(pic_path)
                show_pic(img, coords)  # ԭͼ

                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                cnt += 1

                show_pic(auged_img, auged_bboxes)  # ǿ�����ͼ
