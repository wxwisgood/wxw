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

from  YOLO import YOLO


if __name__ == '__main__':
    model_path = "logs/20181028/yolov3_20181028_6.h5"
    classes_path = 'model_data/object_classes2018_10_26.txt'
    classes_recongition_value_path = 'model_data/object_classes_to_value2018_10_26.txt'
    # anchors_path='yolov3_anchors_416_20181020.txt'
    anchors_path = 'model_data/yolov3_anchors_416_20181020.txt'
    yolo = YOLO(model_image_size=(320, 320), iou=0.2, score=0.1, model_path=model_path,
                classes_path=classes_path, classes_recongition_value_path=classes_recongition_value_path,
                anchors_path=anchors_path, Use_Soft_NMS=False)

    image_path = 'Data\\ball2he\\ball2he11_0005.jpg'
    image_out_path = "image_test/image_out/"
    if(not os.path.exists(image_out_path)):
        os.mkdir(image_out_path)
    # # image = Image.open(image_path).convert('RGB')
    # # image1 = yolo.detect_image(image)
    #
    image_out_name=image_path.split('\\')[-1]
    # image1.save(image_out_path+image_out_name)

    image = Image.open(image_path).convert('RGB')
    score,image2 = yolo.recognition_image(image)
    print(score)
    image2.save(image_out_path +"reco_"+image_out_name)



