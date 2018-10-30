#!/usr/bin/env python
# -*- coding: gbk -*
# -*- coding: utf-8 -*-
# @Time    : 2018/6 
# @Author  : WXW
# @Site    :
# @File    : .py
# @Software: PyCharm

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

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_finally_1.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "classes_recongition_value_path":"model_data/classes_recongition_value.txt",
        "score" : 0.5,
        "iou" : 0.5,#������С�ڴ˲�����Ϊ������һ��Ŀ�ֵ꣬Խ�����Ŀ��Խ�࣬ԽС�����Ŀ��Խ��
        "model_image_size": (288, 288),
        # "model_image_size": (608, 608),
        # "anchors_path": 'model_data/yolov3_anchors_608.txt',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "Use_Soft_NMS":True,
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.class_names_value=self._get_class_value_()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def _get_class_value_(self):
        classes_path = os.path.expanduser(self.classes_recongition_value_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names_value = {}
        for class_value in class_names:
            tep = class_value.split(' ')
            # print(tep)
            class_names_value[tep[0]] = int(tep[1])
        return class_names_value
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def IOU(self,Reframe,GTframe):
        """
        �Զ��庯�������������� IOU������Ϊ��Ϊ���ζԽ��ߣ���x,y��  ���ꡣ
        """
        x1 = Reframe[0]
        y1 = Reframe[1]
        width1 = Reframe[2] - Reframe[0]
        height1 = Reframe[3] - Reframe[1]

        x2 = GTframe[0]
        y2 = GTframe[1]
        width2 = GTframe[2] - GTframe[0]
        height2 = GTframe[3] - GTframe[1]

        endx = max(x1 + width1, x2 + width2)
        startx = min(x1, x2)
        width = width1 + width2 - (endx - startx)

        endy = max(y1 + height1, y2 + height2)
        starty = min(y1, y2)
        height = height1 + height2 - (endy - starty)

        if width <= 0 or height <= 0:
            ratio = 0  # �ص���Ϊ 0
        else:
            Area = width * height  # �������ཻ���
            Area1 = width1 * height1
            Area2 = width2 * height2
            ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio

    def soft_nms(self,boxes, scores, classes, sigma=0.5):

        N = len(boxes)
        for i in range(N):
            # ���ȶ��б��������,����һ�Σ�ִ��һ��
            for j in range(i + 1, N):
                if (scores[i] < scores[j]):
                    scores[i], scores[j] = scores[j], scores[i]
                    #ע��numpy����copy��Ȼֻ��ָ��
                    boxes1, boxes2 = boxes[i].copy(), boxes[j].copy()
                    boxes[i] = boxes2.copy()
                    boxes[j] = boxes1.copy()
                    classes[i], classes[j] = classes[j], classes[i]

            # ����soft�����ֵ����
            for j in range(i + 1, N):
                boxes1, boxes2 = boxes[i], boxes[j]
                # ���㽻����
                ratio = self.IOU(boxes1, boxes2)
                weight = np.exp(-(ratio * ratio) / sigma)
                scores[j] = weight * scores[j]
        # ȥ��ֵ����score��
        boxes1 = []
        scores1 = []
        classes1 = []
        for i in range(N):
            if ( scores[i]> self.score):
                boxes1.append(boxes[i].copy())
                scores1.append(scores[i].copy())
                classes1.append(classes[i].copy())

        return boxes1, scores1, classes1
    def del_boundary_object(self,image,boxes, scores, classes):
        # ����ͼ��߽粢�ҵ÷ֱȽϵͣ�����Ŀ��Ƚ�С���޳�
        Image_Area = image.width * image.height
        i = 0
        while (i < len(classes)):
            predicted_class = self.class_names[classes[i]]
            box = boxes[i]
            score = scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            top, left, bottom, right = box
            area_score = ((right - left) * (bottom - top) / Image_Area)
            boundary_stutas = False  # �Ƿ��ڱ߽���
            boundary_hid = 4  # ��߽������ڱ߽���
            if (top < boundary_hid or bottom > (image.height - boundary_hid)
                    or left < boundary_hid or right > (image.width - boundary_hid)):
                boundary_stutas = True

            # �ж��Ƿ�����ȥ������
            if (area_score < 0.012 and scores[i] < 0.4 and boundary_stutas == True):
                #ɾ��numpy������
                print('del {},({:d},{:d}),({:d},{:d})'.format(label, int(left), int(top),
                                                              int(right),int(bottom)))

                scores = np.delete(scores, i, axis=0)
                boxes=np.delete(boxes, i, axis=0)
                classes=np.delete(classes,i, axis=0)
                # del scores[i]
                # del boxes[i]
                # del classes[i]
            else:
                i = i + 1
        return boxes, scores, classes

    def detect_image_txt(self, image):


        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]]
            })
        if(self.Use_Soft_NMS):
            out_boxes, out_scores, out_classes=self.soft_nms(out_boxes, out_scores, out_classes,sigma=0.5)

        #ȥ���߽��Ͽ��ŶȱȽϵ͵�Ŀ��
        out_boxes, out_scores, out_classes=self.del_boundary_object(image, out_boxes, out_scores, out_classes)




        # else:
        #     out_boxes1, out_scores1, out_classes1=out_boxes, out_scores, out_classes
        return out_boxes, out_scores, out_classes
    #��Ŀ����Ŀ�
    def draw_detect_image(self,image,out_boxes, out_scores, out_classes):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500
        draw = ImageDraw.Draw(image)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            elif(bottom+label_size[1])<image.height:
                text_origin = np.array([left, bottom-1])
            else:
                text_origin = np.array([left, top + 1])
            #�ұ߳�����ʾ������
            if(left+label_size[0]>image.width):
                text_origin[0] = image.width-label_size[0]
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        return image
    #���Ŀ��
    def detect_image(self,image):
        start = timer()
        out_boxes, out_scores, out_classes =self.detect_image_txt(image)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image=self.draw_detect_image(image,out_boxes, out_scores, out_classes)

        end = timer()
        print("Cost time:{}".format(end - start))
        # image.save("E:/1.jpg")
        return image

    #����ֺ�״̬
    def com_object_stutas(self,image,out_boxes, out_scores, out_classes):
        score_fen_he = 0
        Image_Area = image.width * image.height
        score_totals=[]
        if(len(out_boxes)==0):
            score_fen_he=0
            object_stutas=0
            return score_fen_he, object_stutas

        max_score=max(out_scores)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # ���Ȩ��
            area_score = ((right - left) * (bottom - top) / Image_Area) ** (0.3)
            # λ��Ȩ�� Խ�����ӳ�����Ȩ��Խ��
            pianyi_x = abs(((right + left) / 2 - (image.width / 2)) / (image.width / 2))
            pianyi_y = abs(((bottom + top) / 2 - (image.height / 2)) / (image.height / 2))
            position_score=1-max(pianyi_x,pianyi_y)#���ý�������
            # position_score = (1 - (pianyi_x + pianyi_y) / 2) ** (0.9)
            score_totals.append( position_score * area_score * score * self.class_names_value[predicted_class])

        #���Ŀ������Ƚ��٣�ֱ�����
        valid_object_nums=3
        if(len(score_totals)<valid_object_nums):
            score_fen_he = sum(score_totals)
        else:
            #���Ŀ������ر�ֱ࣬ȡǰ������Ч��
            # ð��������������Ч��ǰ����Ŀ��
            for i in range(valid_object_nums):

                for j in range(i+1,len(score_totals)):
                    if(abs(score_totals[i])<abs(score_totals[j])):
                        score_totals[i],score_totals[j]=score_totals[j],score_totals[i]
            #��ֵ
            score_fen_he=sum(score_totals[:valid_object_nums])
        #���max_score�ܸߣ�������ֵ
        if(max_score>0.95):
            score_fen_he_threshold=0.05
        else:
            score_fen_he_threshold=0.1
        # �ж�stutas�Ƿֻ��ߺ�
        if (score_fen_he > score_fen_he_threshold):
            object_stutas = 1
        elif (score_fen_he < -score_fen_he_threshold):
            object_stutas = -1
        else:
            object_stutas = 0
        return score_fen_he,object_stutas

    #��״̬ʶ��Ŀ�
    def draw_recognition_image(self,image,out_boxes, out_scores, out_classes):

        draw = ImageDraw.Draw(image)

        # �жϷֺϵ÷�>0.1 �� <-0.1 �� �м� �޷��ж�
        score_fen_he, object_stutas=self.com_object_stutas(image, out_boxes, out_scores, out_classes)



        label = '{} {:.2f}'.format(object_stutas, score_fen_he)
        if(object_stutas==0):
            text_origin = np.array([20,400])
            color=(0,255,0)
        elif (object_stutas == 1):
            text_origin = np.array([20, 20])
            color = (255, 0, 0)
        else:
            text_origin = np.array([20,740])
            color = (0, 0, 255)
        # ������
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(6e-2 * image.size[1] + 0.5).astype('int32'))
        #��ʮ����
        line_length=100
        thickness=2
        mid_with=int(image.width / 2)
        mid_height=int(image.height / 2)
        #����
        draw.line(
            [mid_with- line_length, mid_height,mid_with+ line_length,
             mid_height], fill=color, width=3)
        draw.line([mid_with,mid_height - line_length, mid_with,
                   mid_height + line_length], fill=color, width=3)

        draw.text(text_origin, label, fill=color, font=font)
        del draw
        return image,score_fen_he, object_stutas

    def recognition_image(self, image):
        start = timer()

        out_boxes,out_scores,out_classes = self.detect_image_txt(image)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image = self.draw_detect_image(image, out_boxes, out_scores,out_classes)
        image, score_fen_he, object_stutas=self.draw_recognition_image(image,out_boxes,out_scores,out_classes)
        end = timer()
        print("Cost time:{}".format(end - start))

        return object_stutas,image



    #���˹���ע��Ϣһ����ʾ
    def recognition_and_drawtest_image(self, image,test_list):
        start = timer()
        set_unpre_name = {'switch', 'knife-switch', 'ground-wire'}
        set_unpre_num = set()
        for i in range(len(self.class_names)):
            if (self.class_names[i] in set_unpre_name):
                set_unpre_num.add(i)

        out_boxes, out_scores, out_classes = self.detect_image_txt(image)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image = self.draw_detect_image(image, out_boxes, out_scores, out_classes)
        image, score_fen_he, object_stutas = self.draw_recognition_image(image, out_boxes, out_scores, out_classes)





        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # ��¼��ʵ��Ϣ
        real_objects = []
        predicted_objects = []


        # ��ʾ��ע��Ϣ
        line = test_list
        a = line.split(' ')
        image_name = a[0]
        len_object = len(a)
        Image_Area = image.width * image.height
        for i in range(1, len_object):
            # print(i)
            object1 = a[i].split(',')
            object2 = [int(tep) for tep in object1]
            print(object2)
            left, top, right, bottom, classname = object2
            real_objects.append(object2)
            classname = self.class_names[classname]

            label = '{}'.format(classname)

            label_size = draw.textsize(label, font)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if bottom + label_size[1] >= image.height:
                text_origin = np.array([left, bottom - label_size[1]])
            else:
                text_origin = np.array([left, bottom + 1])
            # �ұ߳�����ʾ������
            if (left + label_size[0] > image.width):
                text_origin[0] = image.width - label_size[0]
            if (classname not in set_unpre_name):
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=(0, 0, 0))
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=(0, 0, 0))
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        #ͳ��Ԥ����Ϣ
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            predicted_objects.append([left, top, right, bottom, c])

        #�ж��Ƿ�Ԥ�����
        num_FN=0
        #����©���
        for real_object in real_objects:
            classname_real=real_object[4]
            if (classname_real not in set_unpre_num):
                box_real=real_object[0:4]
                if_exist=False
                for predicted_object in predicted_objects:
                    classname_pre=predicted_object[4]
                    box_pre=predicted_object[0:4]
                    if(classname_pre==classname_real):
                        iou=self.IOU(box_real,box_pre)
                        if(iou>0.3):
                            if_exist=True
                            break

                if(if_exist==False):
                    num_FN+=1

        num_FP = 0
        # ����������
        for predicted_object in predicted_objects:
            classname_pre = predicted_object[4]
            if (classname_pre not in set_unpre_num):
                box_pre = predicted_object[0:4]
                if_exist = False
                for real_object in real_objects:
                    classname_real = real_object[4]
                    box_real = real_object[0:4]
                    if (classname_pre == classname_real):
                        iou = self.IOU(box_real, box_pre)
                        if (iou > 0.3):
                            if_exist = True
                            break

                if (if_exist == False):
                    num_FP += 1

        label = 'num_FP:{} num_FN:{}'.format(num_FP, num_FN)
        text_origin = np.array([image.width-500, image.height-100])
        # ������
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))
        draw.text(text_origin, label, fill=(0, 255, 0), font=font)
        if((num_FN+num_FP)>0):
            if_error=True
        else:
            if_error=False

        del draw
        end = timer()
        print(end - start)

        return object_stutas,image,if_error

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
