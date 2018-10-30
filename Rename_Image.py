#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/10
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

import os
Image='arrow2he'
pici="1"
pici2=1
input_dir="D:/WXW/1014/"+Image
image_names=os.listdir(input_dir)
for image_name in image_names:
    image1=image_name.split('_')[-1]
    pici2 = 1

    output=Image+pici+"_"+str(pici2)+'_'+image1
    while(os.path.exists(os.path.join(input_dir,output))):
        pici2=pici2+1
        output = Image + pici + "_" + str(pici2) + '_' + image1

    print(image_name)
    print(output)
    os.rename(os.path.join(input_dir,image_name),os.path.join(input_dir,output))