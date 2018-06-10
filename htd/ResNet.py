import keras
import argparse
import math
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser(description='Residual Network Training code(Keras version)')
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
                help='dataset. (default: cifar10)')
parser.add_argument('-s','--scheduler', type=str, default="step_decay", metavar='STRING',
                help='learning rate scheduler. [step_decay, cos or tanh] (default: step_decay)')
parser.add_argument('-c','--count', type=int, default=1, metavar='NUMBER',
                help='runs number. (default: 1)')

args = parser.parse_args()

stack_n            = args.stack_n
batch_size         = args.batch_size
epochs             = args.epochs
scheduler          = args.scheduler
layers             = 6 * stack_n + 2
img_rows, img_cols = 32, 32
img_channels       = 3 
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

# param for cos & tanh
start_lr           = 0.1
end_lr             = 0.0

# learning rate scheduler
# step_decay , cos and tanh
def step_decay_scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001

def cos_scheduler(epoch):
    return (start_lr+end_lr)/2.+(start_lr-end_lr)/2.*math.cos(math.pi/2.0*(epoch/(epochs/2.0)))

def tanh_scheduler(epoch):
    start = -6.0
    end = 3.0
    return start_lr / 2.0 * (1- math.tanh( (end-start)*epoch/epochs + start))

class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def print_best_acc(self):
        print('== BEST ACC: {:.4f} =='.format(max(self.val_acc['epoch'])))

# residual networks
def residual_network(img_input,classes_num=10,stack_n=5):
    
    def bn_relu(x):
    	x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    	return Activation('relu')(x)
    def conv(x,out,size,stride):
    	return Conv2D(out,kernel_size=size,strides=stride,
    		   padding='same',
    		   kernel_initializer="he_normal",
    		   kernel_regularizer=regularizers.l2(weight_decay)
    		   )(x)

    def residual_block(x,o_filters,increase=False):
        stride = (2,2) if increase else (1,1)
     
        o1     = bn_relu(x)
        conv_1 = conv(o1,o_filters,(3,3),stride) 
        o2     = bn_relu(conv_1)
        conv_2 = conv(o2,o_filters,(3,3),(1,1))
     
        if increase:
            projection = conv(o1,o_filters,(1,1),(2,2))
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    out = Dense(classes_num,activation='softmax',
    	      kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return out

if __name__ == '__main__':

    print("\n========================================") 
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2)) 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))
    print("LEARNING RATE SCHEDULER: {:}".format(args.scheduler))


    print("\n== LOADING DATA... ==")
    
    # load data cifar-10 or cifar-100
    if args.dataset == "cifar100":
        num_classes = 100
        mean = [129.3, 124.1, 112.4]
        std  = [68.2, 65.4, 70.4]
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        num_classes = 10
        mean = [125.3, 123.0, 113.9]
        std  = [63.0, 62.1, 66.7]
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("\n== DONE! ==\n\n== COLOR PREPROCESSING... ==")
   
    # color preprocessing
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    print("\n== DONE! ==\n\n== BUILD MODEL... ==")
   
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output)
    
    # print model architecture if you need.
    # print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    if scheduler == 'step_decay':
         lrs = LearningRateScheduler(step_decay_scheduler)
    elif scheduler == 'cos':
         lrs = LearningRateScheduler(cos_scheduler)
    elif scheduler == 'tanh':
         lrs = LearningRateScheduler(tanh_scheduler)

    tb   = TensorBoard(log_dir='./resnet_{:d}_{}_{}_{}/'.format(layers,args.dataset,scheduler,args.count), histogram_freq=0)
    ckpt = ModelCheckpoint('resnet_{:d}_{}_{}_{}.h5'.format(layers,args.dataset,scheduler,args.count), 
                            monitor='val_acc', verbose=0, save_best_only=True, mode='auto', period=1)
    his  = AccHistory()
    cbks = [tb,lrs,ckpt,his]
    
    # set data augmentation
    print("\n== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    his.print_best_acc()
    print("\n== FINISH TRAINING! ==") 