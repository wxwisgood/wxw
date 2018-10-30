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


from keras.utils import multi_gpu_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from  YOLO_tiny import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__=='__main__':
    yolo=YOLO()
    image_path='image_test'

    image_out=image_path + '/out4/'
    if(not os.path.exists(image_out)):
        os.mkdir(image_out)
    image_path_names=os.listdir(image_path)
    for imagename in image_path_names:
        if( imagename.endswith('.jpg')):
            image_in=image_path+'/'+imagename
            print(image_in)
            image_out_name=image_out+imagename
            image = Image.open(image_in).convert('RGB')
            score,image1 = yolo.recognition_image(image)
            image1.save(image_out_name)

