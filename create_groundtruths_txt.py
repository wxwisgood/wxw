#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/6 
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

import os
object_txt='Data/test.txt'
class_txt='Data/object_classes2018_10_29.txt'
output_txt='mAP/Object-Detection-Metrics-master/groundtruths/'\

with open(class_txt) as f:
    class_names = f.readlines()
for i in range(len(class_names)):
    class_names[i]=class_names[i][:-1]

with open(object_txt) as f:
    lines = f.readlines()
for line in  lines:
    line=line[:-1]
    a=line.split(' ')
    image_name=a[0]

    image_name1=(image_name.split('\\')[-1]).split('.')[0]
    if (image_name1 == '29'):
        dd = 1
    with open((output_txt+image_name1+".txt"),'w+') as f1:

        for i in range(1,len(a)):
            object_all = a[i]
            object_all=object_all.split(',')
            xmin=object_all[0]
            ymin = object_all[1]
            xmax = object_all[2]
            ymax = object_all[3]
            object_class_name=class_names[int(object_all[4])]
            f1.write(object_class_name)
            f1.write(' ')
            f1.write(xmin)
            f1.write(' ')
            f1.write(ymin)
            f1.write(' ')
            f1.write(xmax)
            f1.write(' ')
            f1.write(ymax)
            f1.write(' ')
            f1.write('\n')





