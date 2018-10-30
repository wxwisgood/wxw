# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from yolo3.model_mutiGPU import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from keras.utils import multi_gpu_model
from yolo3.parallel_model import ParallelModel,ParallelModelCheckpoint
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _main():
    annotation_path_train = 'Data\\train.txt'
    annotation_path_val = 'Data\\val.txt'
    log_dir = 'logs/20181024/'
    classes_path = 'model_data/object_classes2018_10_24.txt'
    anchors_path = 'model_data/yolov3_anchors_416_20181020.txt'
    class_names = get_classes(classes_path)
    num_classes=len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,weights_path='logs/20181022/yolov3_20181022_4.h5')
    train_num = 0
    model_name='yolov3_20181024_'
    save_model_name = model_name+ str(train_num) + '.h5'

    train(model, annotation_path_train,annotation_path_val, input_shape, anchors, len(class_names),save_model_name,log_dir=log_dir)

    #第二阶段多尺度训练
    muti_scales=[(416,416),(608,608),(320,320),(416,416),(608,608),(320,320),(608,608)]
    # muti_scales = [(256, 256), (288, 288), (416, 416), (608, 608), (288, 288)]



    for input_shape in muti_scales:
        open_model_name=save_model_name
        model = create_model(input_shape, anchors, num_classes,freeze_body=False, weights_path=log_dir+open_model_name)
        train_num = train_num + 1
        save_model_name = model_name+ str(train_num) + '.h5'
        train_mutiscale(model, annotation_path_train, annotation_path_val, input_shape, anchors, len(class_names),save_model_name,log_dir=log_dir)






def train(model, annotation_path_train,annotation_path_val, input_shape, anchors, num_classes,save_model_name,log_dir='logs/'):
    p_model = multi_gpu_model(model, gpus=2)
    print(len(p_model.layers))
    print(len(model.layers))
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ParallelModelCheckpoint(model,log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                         monitor='val_loss', save_weights_only=True, save_best_only=True, period=10)
    # checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
    #                              monitor='val_loss', save_weights_only=True, save_best_only=True, period=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=2)  # 当评价指标不在提升时，减少学习率
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2)  # 验证集准确率，下降前终止
    batch_size = 32
    p_model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred[0]})

    with open(annotation_path_train) as f:
        lines_train = f.readlines()
    np.random.shuffle(lines_train)

    with open(annotation_path_val) as f:
        lines_val = f.readlines()
    np.random.shuffle(lines_val)

    num_val = len(lines_val)
    num_train = len(lines_train)
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    p_model.fit_generator(data_generator_wrap(lines_train, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrap(lines_val, batch_size, input_shape, anchors,
                                                            num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint])
    model.save_weights(log_dir + save_model_name)


def train_mutiscale(model, annotation_path_train, annotation_path_val, input_shape, anchors, num_classes,save_model_name,log_dir='logs/'):
    p_model = multi_gpu_model(model, gpus=2)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ParallelModelCheckpoint(model, log_dir +str(input_shape[0])+"_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                         monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # checkpoint = ModelCheckpoint(log_dir +str(input_shape[0])+"_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
    #                              monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=2)  # 当评价指标不在提升时，减少学习率
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=2)  # 验证集准确率，下降前终止 patience次后中止


    with open(annotation_path_train) as f:
        lines_train = f.readlines()
    np.random.shuffle(lines_train)

    with open(annotation_path_val) as f:
        lines_val = f.readlines()
    np.random.shuffle(lines_val)

    num_val = len(lines_val)
    num_train = len(lines_train)




    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    # p_model.compile(optimizer='adam', loss={
    #     'yolo_loss': lambda y_true, y_pred: y_pred[0]})
    p_model.compile(optimizer=Adam(lr=1e-4), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred[0]})
    if(input_shape[0]<400):
        batch_size = 32  #288
    elif(input_shape[0]>600):
        batch_size = 8
    else:
        batch_size = 16

    # if (input_shape == (288, 288)):
    #     batch_size = 24
    # elif (input_shape == (256, 256)):
    #     batch_size = 28
    # elif (input_shape == (416, 416)):
    #     batch_size = 12
    # elif (input_shape == (608, 608)):
    #     batch_size = 6
    print('Train on {} samples, val on {} samples, with batch size {} input_shape {}.'.format(num_train, num_val,batch_size,
                                                                                              input_shape))
    p_model.fit_generator(data_generator_wrap(lines_train, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrap(lines_val, batch_size, input_shape, anchors,
                                                            num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=80,
                        initial_epoch=50,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + save_model_name)


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True,
                 weights_path='model_data/yolo_weights.h5'):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            #20181002 0100 修改 laster three layer are conv2D 因此从-7 改为-3
            num = len(model_body.layers) - 3
            print(num)
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()