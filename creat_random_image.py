#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/6 
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm




from PIL import Image, ImageFilter
import colorsys
import os
import numpy as np
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
from yolo3.utils import get_random_data
from scipy import misc
from PIL import Image, ImageFont, ImageDraw
import time
import numpy as np

annotation_path_train='Data\\test.txt'
with open(annotation_path_train) as f:
    lines = f.readlines()
input_shape=(416,416)

image_path = 'image_test'

image_out = image_path + '/random416/'
if (not os.path.exists(image_out)):
    os.mkdir(image_out)
classes_path = 'Data/object_classes2018_10_29.txt'
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
def draw_ori_image(image_name,class_names,colors):
    image = Image.open(image_name).convert('RGB')

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 500

    draw = ImageDraw.Draw(image)
    # 显示标注信息

    len_object = len(a)
    Image_Area = image.width * image.height
    for i in range(1, len_object):
        # print(i)
        object1 = a[i].split(',')
        object2 = [int(tep) for tep in object1]
        print(object2)
        left, top, right, bottom, classname1 = object2

        classname = class_names[classname1]

        label = '{}'.format(classname)

        label_size = draw.textsize(label, font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(label, (left, top), (right, bottom))

        if bottom + label_size[1] >= image.height:
            text_origin = np.array([left, bottom - label_size[1]])
        else:
            text_origin = np.array([left, bottom + 1])
        # 右边超出显示往左移
        if (left + label_size[0] > image.width):
            text_origin[0] = image.width - label_size[0]

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[classname1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[classname1])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    return  image

def draw_random_image(image_name,class_names,colors):
    image_new = Image.open(image_name).convert('RGB')
    image=image_new
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 500
    draw = ImageDraw.Draw(image_new)
    len_object = len(box)

    for i in range(0, len_object):
        # print(i)

        object2 = box[i]
        # print(object2)
        left, top, right, bottom, classname = object2
        if(not(left==0 and top==0 and bottom==0 and right==0)):
            classname1=int(classname)
            classname = class_names[classname1]

            label = '{}'.format(classname)

            label_size = draw.textsize(label, font)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))


            if bottom + label_size[1] >= image.height:
                text_origin = np.array([left, bottom - label_size[1]])
            else:
                text_origin = np.array([left, bottom + 1])
            # 右边超出显示往左移
            if (left + label_size[0] > image.width):
                text_origin[0] = image.width - label_size[0]

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[classname1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[classname1])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    return image
sum_time=0
# for i in range(len(lines)):
for i in np.arange(0,len(lines),3):

# for i in range(1000):

    line = lines[i]
    line=line[:-1]
    a = line.split(' ')
    image_name = a[0]
    image_name1 = (image_name.split('\\')[-1]).split('.')[0]
    # print(image_name)
    image_out_name = image_out + image_name1 + '_1.jpg'
    # image=draw_ori_image(image_name, class_names, colors)
    # image.save(image_out_name)


    time1=time.time()
    image1, box = get_random_data(lines[i], input_shape, random=True)
    sum_time=sum_time+(time.time() - time1)

    # image_out_name = image_out + 'temp.jpg'
    image_out_name = image_out + image_name1 + '_2.jpg'
    # 显示标注信息
    misc.imsave(image_out_name, image1)

    image=draw_random_image(image_out_name, class_names, colors)
    # image_out_name = image_out + image_name1 + '_2.jpg'
    # image_new.save(image_out_name)
    # # misc.imsave(image_out_name, image)
    # # image = Image.fromarray(image)
    image.save(image_out_name)
    sum_time = sum_time + (time.time() - time1)
print(sum_time)