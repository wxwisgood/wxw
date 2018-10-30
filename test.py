#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/6 
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm
import os

import matplotlib.pyplot as plt
import numpy as np

classes_path='model_data/yolo_anchors.txt'
with open(classes_path) as f:
    class_names = f.readlines()
print(class_names)
a=class_names[0][:-1]
b=a.split(",")
c=[]
for i in b:
    c.append(int(int(i)/416*608))
print(c)

f = open("yolov3_anchors_608.txt", 'w')
for i in range(int(len(c)/2)):
    if i == 0:
        x_y = "%d,%d" % (c[2*i], c[2*i+1])
    else:
        x_y = ", %d,%d" % (c[2*i], c[2*i+1])
    f.write(x_y)
f.close()


