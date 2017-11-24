# -*- coding:utf-8 -*-  

import os
import sys
import time
import pickle
import random
import math
import numpy as np

class_num       = 10
image_size      = 32
img_channels    = 3


# ========================================================== #
# ├─ prepare_data() 
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #



def download_data():
    dirname  = 'cifar-10-batches-py'
    origin   = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = 'cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet aready exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count*block_size*100/total_size),100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                            (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file):
    batch  = unpickle(file)
    data   = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." %(file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data,data_n,axis=0)
        labels = np.append(labels,labels_n,axis=0)
    labels = np.array( [ [ float( i == label ) for i in range(label_count) ] for label in labels ] )
    data = data.reshape([-1,img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle( data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = [ 'data_batch_%d' % d for d in range(1,6) ]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)

    print("Train data:",np.shape(train_data), np.shape(train_labels))
    print("Test data :",np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


# ========================================================== #
# ├─ _random_crop() 
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        
        if padding:
            oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                        nw:nw + crop_shape[1]]
        return new_batch

def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    return batch
