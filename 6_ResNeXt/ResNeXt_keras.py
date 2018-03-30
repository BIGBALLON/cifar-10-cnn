import keras
import math
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.regularizers import l2

CARDINALITY        = 8            # 4 or 8 or 16
BASE_WIDTH         = 64
IN_PLANES          = 64

IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 3
CLASS_NUM          = 10
BATCH_SIZE         = 64           # 32 or 64 or 128
epochs             = 300
ITERATIONS         = 50000 // BATCH_SIZE + 1
WEIGHT_DECAY       = 5e-4

mean = [125.3, 123.0, 113.9]
std  = [63.0,  62.1,  66.7]

from keras import backend as K

# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

def resnext(img_input,classes_num):
    global IN_PLANES
    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x,planes,stride):
        h = planes // CARDINALITY
        groups = []
        for i in range(CARDINALITY):
            group = Lambda(lambda z: z[:,:,:, i * h : i * h + h])(x)
            groups.append(Conv2D(h,kernel_size=(3,3),strides=stride,kernel_initializer="he_normal",
                kernel_regularizer=l2(WEIGHT_DECAY),
                padding='same',use_bias=False)(group))
        x = concatenate(groups)
        return x

    def residual_block(x,planes,stride=(1,1)):

        D = int(math.floor(planes * (BASE_WIDTH/64.0)))
        C = CARDINALITY

        shortcut = x
        
        y = Conv2D(D*C,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer="he_normal",kernel_regularizer=l2(WEIGHT_DECAY),use_bias=False)(shortcut)
        y = bn_relu(y)

        y = group_conv(y,D*C,stride)
        y = bn_relu(y)

        y = Conv2D(planes * 4, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer="he_normal",kernel_regularizer=l2(WEIGHT_DECAY),use_bias=False)(y)
        y = bn_relu(y)

        if stride != (1,1) or IN_PLANES != planes * 4:
            shortcut = Conv2D(planes * 4, kernel_size=(1,1), strides=stride, padding='same', kernel_initializer="he_normal",kernel_regularizer=l2(WEIGHT_DECAY),use_bias=False)(x)
            shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)
        
        y = add([y,shortcut])
        y = Activation('relu')(y)
        return y
    
    def residual_layer(x, blocks, planes, stride=(1,1)):
        x = residual_block(x, planes, stride)
        IN_PLANES = planes * 4
        for i in range(1,blocks):
            x = residual_block(x,planes)
        return x
    
    def conv3x3(x,filters):
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer="he_normal",kernel_regularizer=l2(WEIGHT_DECAY),use_bias=False)(x)
        x = bn_relu(x)
        return x

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer="he_normal",kernel_regularizer=l2(WEIGHT_DECAY),use_bias=False)(x)


    # build the resnext model    
    x = conv3x3(img_input,64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128,stride=(2,2))
    x = residual_layer(x, 3, 256,stride=(2,2))
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_test  = keras.utils.to_categorical(y_test, CLASS_NUM)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))
    output    = resnext(img_input,CLASS_NUM)
    resnet    = Model(img_input, output)

    print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb     = TensorBoard(log_dir='./resnext/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint('./ckpt.h5', save_best_only=False, mode='auto', period=25)
    cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE), steps_per_epoch=ITERATIONS, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
    resnet.save('resnext.h5')
