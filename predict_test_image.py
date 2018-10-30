# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os

from keras.utils import multi_gpu_model
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from  YOLO import YOLO
# from  YOLO_tiny import YOLO

if __name__=='__main__':
    # model_path = "logs/20181024_1080Ti/yolov3_20181024_2.h5"
    model_path = "logs/20181026/yolov3_20181026_3.h5"
    classes_path ='model_data/object_classes2018_10_26.txt'
    classes_recongition_value_path='model_data/object_classes_to_value2018_10_26.txt'
    # anchors_path='yolov3_anchors_416_20181020.txt'
    anchors_path = 'model_data/yolov3_anchors_416_20181020.txt'
    yolo=YOLO(model_image_size=(416,416),iou=0.2,score=0.03,model_path=model_path,
              classes_path=classes_path,classes_recongition_value_path=classes_recongition_value_path,
              anchors_path=anchors_path,Use_Soft_NMS=False)

    object_txt = 'Data\\test.txt'

    with open(object_txt) as f:
        lines = f.readlines()

    image_path = 'image_test'

    image_out = image_path + '/test_image_20181026/'
    if (not os.path.exists(image_out)):
        os.mkdir(image_out)


    for line in lines:
        line = line[:-1]
        a = line.split(' ')
        image_name = a[0]

        print(image_name)

        image = Image.open(image_name).convert('RGB')
        image_name1 = (image_name.split('\\')[-1]).split('.')[0]
        image_out_name=image_out+image_name1+'.jpg'
        score,image1 = yolo.recognition_image(image)

        image1.save(image_out_name)

