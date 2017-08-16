import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, LeakyReLU, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers
from keras import regularizers

cardinality        = 16

num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 391
weight_decay       = 0.0001

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
lr   = [0.1, 0.02, 0.004, 0.0008]

def scheduler(epoch):
    return lr[epoch // 55]

def resnext(img_input,classes_num):
    def add_common_layer(x):
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def group_conv(x,out_filters,strides):
        if cardinality == 1:
            return Conv2D(out_filters,kernel_size=(3,3),strides=strides,padding='same')(x)
        
        h = out_filters // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:,:,:, i * h : i * h + h])(x)
            groups.append(Conv2D(h,kernel_size=(3,3),strides=strides,padding='same')(group))
        x = concatenate(groups)
        return x

    def residual_block(x,in_filters,out_filters,increase_filter=False):
        first_stride = (1,1)
        if increase_filter:
            first_stride = (2,2)
        shortcut = x
        y = add_common_layer(shortcut)
        
        y = Conv2D(in_filters,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(y)
        y = add_common_layer(y)

        y = group_conv(y,in_filters,strides=first_stride)
        y = add_common_layer(y)

        y = Conv2D(out_filters, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(y)

        if increase_filter or in_filters != out_filters:
            shortcut = Conv2D(out_filters,kernel_size=(1,1),strides=first_stride,padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
        
        y = add([y,shortcut])
        return y
    
    def residual_layer(x,num,in_filters,out_filters,increase_filter=False):
        x = residual_block(x,in_filters,out_filters,increase_filter)
        for _ in range(1,num):
            x = residual_block(x,in_filters,out_filters)
        return x        
    
    def conv3x3(x,filters):
        return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)


    # build the resnext model    
    x = conv3x3(img_input,64)
    x = residual_layer(x, 3, 64, 128)
    x = residual_layer(x, 3, 128, 256,increase_filter=True)
    x = residual_layer(x, 3, 256, 512,increase_filter=True)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output = resnext(img_input,num_classes)
    resnet = Model(img_input, output)
    print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb = TensorBoard(log_dir='./resnext/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
    resnet.save('resnext.h5')