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

from  YOLO_tiny import YOLO


if __name__=='__main__':
    model_path = "logs/201810182/trained_weights_finally_40.h5"
    classes_path = 'Data/object_classes2018_10_18.txt'
    classes_recongition_value_path = 'Data/object_classes_to_value2018_10_18.txt'

    yolo = YOLO(model_image_size=(608, 608), iou=0.2, score=0.5, model_path=model_path,
                classes_path=classes_path, classes_recongition_value_path=classes_recongition_value_path)

    object_txt = 'Data\\all_train_data.txt'

    with open(object_txt) as f:
        lines = f.readlines()

    image_path = 'image_test'

    image_out = image_path + '/all_real_tiny/'
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
        # score,image1 = yolo.recognition_image(image)
        score,image1,if_error= yolo.recognition_and_drawtest_image(image,line)
        if(if_error==True):
            image1.save(image_out_name)

