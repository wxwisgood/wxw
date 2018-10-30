#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/10
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

import cv2
import time
from scipy.misc import imread
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import io


object_txt = 'Data\\test.txt'
with open(object_txt) as f:
    lines = f.readlines()
lines=lines[:1]
time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('opencv cost time:{}'.format(time.time() - time1))

time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = Image.open(image_name)
    img = np.array(img)
print('PIL cost time:{}'.format(time.time() - time1))

time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = Image.open(image_name)
    # img = np.array(img)
print('PIL1 cost time:{}'.format(time.time() - time1))

time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = tf.gfile.FastGFile(image_name, 'rb').read()
    img = tf.image.decode_jpeg(img)
print('tf cost time:{}'.format(time.time() - time1))

time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = io.imread(image_name)
print('skimage cost time:{}'.format(time.time() - time1))


time1 = time.time()
for line in lines:
    line = line[:-1]
    a = line.split(' ')
    image_name = a[0]
    img = imread(image_name)

print('scipy cost time:{}'.format(time.time() - time1))

