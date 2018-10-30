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

from  YOLO_tiny import YOLO
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    yolo = YOLO()


    image_path = 'image_test/raccoon-150.jpg'
    image = Image.open(image_path).convert('RGB')
    image1 = yolo.detect_image(image)
    image1.save("image_test/out2/raccoon-150.jpg")





