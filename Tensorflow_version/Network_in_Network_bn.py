# ===================================================================== #
# File name:           Network_in_Network.py
# Author:              BIGBALLON
# Date update:         07/28/2017
# Python Version:      3.5.2
# Tensorflow Version:  1.2.1
# Description: 
# Dataset:             Cifar-10
# Testing accuracy:    91.5%
# ===================================================================== #

import tensorflow as tf
from data_utility import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_save_path', './nin_logs', 'Directory where to save tensorboard log')
tf.app.flags.DEFINE_string('model_save_path', './model/', 'Directory where to save model weights')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('iteration', 391, 'iteration')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
tf.app.flags.DEFINE_float('epochs', 164, 'epochs')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')

# ========================================================== #
# ├─ conv()
# ├─ activation(x)
# ├─ max_pool()
# └─ global_avg_pool()
# ========================================================== #

def conv(x, is_train, shape):
    he_initializer = tf.contrib.keras.initializers.he_normal()
    W = tf.get_variable('weights', shape=shape, initializer=he_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=is_train, updates_collections=None)    

def activation(x):
    return tf.nn.relu(x) 

def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def learning_rate_schedule(epoch_num):
      if epoch_num < 81:
          return 0.085
      elif epoch_num < 121:
          return 0.01
      else:
          return 0.001

def main(_):
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    # define placeholder x, y_ , keep_prob, learning_rate
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, class_num], name='input_y')
        use_bn = tf.placeholder(tf.bool, name='phase')

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('learning_rate'):
        learning_rate = tf.placeholder(tf.float32)

    # build_network

    with tf.variable_scope('conv1'):
        output = conv(x, use_bn, [5, 5, 3, 192])
        output = activation(output)

    with tf.variable_scope('mlp1-1'):
        output = conv(output, use_bn, [1, 1, 192, 160])
        output = activation(output)

    with tf.variable_scope('mlp1-2'):
        output = conv(output, use_bn, [1, 1, 160, 96])
        output = activation(output)

    with tf.name_scope('max_pool-1'):
        output  = max_pool(output, 3, 2)

    with tf.name_scope('dropout-1'):
        output = tf.nn.dropout(output,keep_prob)

    with tf.variable_scope('conv2'):
        output = conv(output, use_bn, [5, 5, 96, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-2'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.name_scope('max_pool-2'):
        output  = max_pool(output, 3, 2)

    with tf.name_scope('dropout-2'):
        output = tf.nn.dropout(output,keep_prob)

    with tf.variable_scope('conv3'):
        output = conv(output, use_bn, [3, 3, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-2'):
        output = conv(output, use_bn, [1, 1, 192, 10])
        output = activation(output)

    with tf.name_scope('global_avg_pool'):
        output  = global_avg_pool(output, 8, 1)

    with tf.name_scope('moftmax'):
        output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # weight decay: l2 * WEIGHT_DECAY
    # train_step: training operation

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

    with tf.name_scope('l2_loss'):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    with tf.name_scope('train_step'):
        train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum,use_nesterov=True).minimize(cross_entropy + l2 * FLAGS.weight_decay)

    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()
    
    # for testing
    def run_testing(sess):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = 1000
        for it in range(10):
            batch_x = test_x[pre_index:pre_index+add]
            batch_y = test_y[pre_index:pre_index+add]
            pre_index = pre_index + add
            loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, use_bn: False,keep_prob: 1.0})
            loss += loss_ / 10.0
            acc += acc_ / 10.0
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss), 
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
        return acc, loss, summary

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.log_save_path,sess.graph)

        # epoch = 164 
        # batch size = 128
        # iteration = 391
        # we should make sure [bath_size * iteration = data_set_number]

        for ep in range(1,FLAGS.epochs+1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\nepoch %d/%d:" %(ep,FLAGS.epochs))

            for it in range(1,FLAGS.iteration+1):
                if pre_index+FLAGS.batch_size < 50000:
                    batch_x = train_x[pre_index:pre_index+FLAGS.batch_size]
                    batch_y = train_y[pre_index:pre_index+FLAGS.batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]


                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],feed_dict={x:batch_x, y_:batch_y, use_bn: True,keep_prob: FLAGS.dropout, learning_rate: lr})
                batch_acc = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, use_bn: True, keep_prob: 1.0})

                train_loss += batch_loss
                train_acc  += batch_acc
                pre_index  += FLAGS.batch_size

                if it == FLAGS.iteration:
                    train_loss /= FLAGS.iteration
                    train_acc /= FLAGS.iteration

                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss), 
                                          tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, FLAGS.iteration, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" %(it, FLAGS.iteration, train_loss / it, train_acc / it) , end='\r')

        save_path = saver.save(sess, FLAGS.model_save_path)
        print("Model saved in file: %s" % save_path)        



if __name__ == '__main__':
    tf.app.run()

    



          
