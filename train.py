"""
Retrain the YOLO model for your own dataset.
"""

import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, luo_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from yolo3.parallel_model import ParallelModel,ParallelModelCheckpoint

def _main():
    '''加载训练数据的图片、bbox坐标、类别ID'''
    annotation_path = 'train.txt'
    log_dir = 'logs/002/'
    '''加载先验框的大小、类别的名称'''
    classes_path = 'model_data/luo_ruizhi_classes.txt'
    anchors_path = 'model_data/luo_ruizhi_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    '''设置输入的大小'''
    input_shape = (416,416) # multiple of 32, hw

    '''判断使用的yolo的版本，是完整版还是精简版'''
    # is_tiny_version = len(anchors)==6 # default setting
    # if is_tiny_version:
    #     model = create_tiny_model(input_shape, anchors, num_classes,
    #         freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    # else:
    #     model = create_model(input_shape, anchors, num_classes,
    #         freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
    '''rewrite'''
    with tf.device('/cpu:0'):
        if len(anchors) == 6:
            model = create_tiny_model(input_shape, anchors, num_classes,
                                      freeze_body=2, weights_path='model_data/tiny_yolo_weight.h5')
        elif len(anchors) == 9:
            model = create_model(input_shape, anchors, num_classes,
                                 freeze_body=2,
                                 weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze
        else:
            model = create_luo_model(input_shape, anchors, num_classes,
                                     freeze_body=2, weights_path='model_data/yolo_weights.h5')

    '''将模型在多张GPU上数据并行'''
    p_model = multi_gpu_model(model,gpus=2)
    '''将模型的结构打印出来'''
    plot_model(model,to_file='logs/000/model.png',show_shapes=True)
    '''设置tensorboard和checkpoint，学习速率衰减以及早终结'''
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ParallelModelCheckpoint(model,log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    '''设置验证样本比例，以及打乱样本顺序'''
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    '''在冻结某些层的网络上训练一次，得到稳定的loss'''
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        p_model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred[0]})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        p_model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    '''解冻所有层，继续训练，来微调'''
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        p_model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred[0]}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        p_model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=450,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    '''多层输出，写在一个列表里，注意那是字典及索引'''
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    '''这个body模型很直接的生成了图片作为输入，anchor作为输出的模型'''
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        '''载入相同名称的部分权重，打印跳过的层'''
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            '''设置部分层不可训练'''
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    '''添加Lambda层，对yolo的输出计算loss'''
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    '''将模型的输出改为loss后构建模型，注意这里的星号不是指指针，是一种多变量输入的用法，与*args一致'''
    model = Model([model_body.input, *y_true], model_loss)
    model.summary()

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
    model.summary()

    return model

'''增加只输出一个尺度的模型'''
def create_luo_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/luo_yolo_weights.h5'):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = Input(shape=(h//32,w//32,num_anchors,num_classes+5))

    model_body = luo_yolo_body(image_input, num_anchors, num_classes)
    print('Create Luo YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet53 body or freeze all but 1 output layers.
            num = (185, len(model_body.layers) - 1)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.8})(
        [model_body.output, y_true])
    model = Model([model_body.input, y_true], model_loss)
    model.summary()

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    _main()
