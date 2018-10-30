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
    yolo=YOLO()

    object_txt = 'VOCdevkit\\VOC2007\\2007_test.txt'
    class_txt = 'model_data/voc_classes.txt'

    with open(class_txt) as f:
        class_names = f.readlines()
    for i in range(len(class_names)):
        class_names[i] = class_names[i][:-1]

    with open(object_txt) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        a = line.split(' ')
        image_name = a[0]

        print(image_name)

        image = Image.open(image_name)
        out_boxes, out_scores, out_classes = yolo.detect_image_txt(image)


        image_name1=(image_name.split('/')[-1]).split('.')[0]

        with open(('mAP/Object-Detection-Metrics-master/detections/'+image_name1+".txt"),'w+') as f1:
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class =class_names[c]
                box = out_boxes[i]
                score = str(out_scores[i])
                ymin, xmin, ymax, xmax = box
                xmin=int(xmin)
                xmin=str(int(xmin));ymin=str(int(ymin));xmax=str(int(xmax));ymax=str(int(ymax))
                f1.write(predicted_class)
                f1.write(' ')
                f1.write(score[1:3])
                f1.write(' ')

                f1.write(xmin)
                f1.write(' ')
                f1.write(ymin)
                f1.write(' ')
                f1.write(xmax)
                f1.write(' ')
                f1.write(ymax)
                f1.write(' ')

