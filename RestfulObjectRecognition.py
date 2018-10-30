#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/6 
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm



import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from PIL import Image, ImageFont, ImageDraw



from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from  YOLO import YOLO




image_out_path='image_test\\'
if(not os.path.exists(image_out_path)):
    os.mkdir(image_out_path)


TODOS =  {'SwitchName': '901 开关','SwitchStatus':'开'}

def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


# # 操作（put / get / delete）单一资源Todo 获取所有的信息
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    #获取所有的信息

    def get(self):
        return TODOS
    def post(self):
        global start
        global status
        global yolo
        args = parser.parse_args()
        print(args)
        ImageName=args['ImageName']
        SwitchName=args['SwitchName']
        SwitchStatus=args['SwitchStatus']

        TODOS['SwitchName']=SwitchName
        TODOS['SwitchStatus']= SwitchStatus

        try:
            # print(ImageName)
            image = Image.open(ImageName)
            object_stutas,image = yolo.recognition_image(image)
            dirname, basename = os.path.split(ImageName)

            image_out_name=image_out_path+basename

            image_out_name = image_out_name[:-4] + '.png'
            print(image_out_name)
            print(object_stutas)
            TODOS['SwitchStatus'] = object_stutas
            image.save(image_out_name)
        except:
            print("no image")


        return TODOS, 201






# 这行代码是只让模式初始化一次
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    model_path='model_data/trained_weights_finally_20101018.h5'
    classes_path = 'model_data/object_classes2018_10_18.txt'
    classes_recongition_value_path = 'model_data/object_classes_to_value2018_10_18.txt'

    yolo = YOLO(model_image_size=(288, 288), iou=0.2, score=0.3, model_path=model_path,
                classes_path=classes_path, classes_recongition_value_path=classes_recongition_value_path)



    #为了解决keras 在Flask下报错，在初始化之后先运行一次程序
    image = Image.open('image_test/1.jpg')
    object_stutas,image1 = yolo.recognition_image(image)
    # print(image1)


if __name__ == '__main__':
    parser = reqparse.RequestParser()
    parser.add_argument('ImageName')
    parser.add_argument('SwitchName')
    parser.add_argument('SwitchStatus')
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Todo, '/todos')
    app.run(host="0.0.0.0", port=8383,debug=True)


