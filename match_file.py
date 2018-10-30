#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/10
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

import  os
input1='D:\WXW\python\Python36_OpenCV\work\YOLOv3_keras\VOCdevkit\VOC2007\Annotations'
input2='D:\WXW\python\Python36_OpenCV\work\YOLOv3_keras\VOCdevkit\VOC2007\JPEGImages'

input1_names=os.listdir(input1)
input2_names=os.listdir(input2)
n1=[a.split('.')[0] for a in input1_names]
n1=set(n1)
n2=[a.split('.')[0] for a in input2_names]
n3=set(n2)
print(len(n1))
print(len(n2))
print(len(n3))
#寻找不匹配的
for name in n1:
    if(not(name in n3)):
        print(name)

