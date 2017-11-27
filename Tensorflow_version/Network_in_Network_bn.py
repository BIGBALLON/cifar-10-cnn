import tensorflow as tf
from data_utility import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_save_path', './nin_logs', 'Directory where to save tensorboard log')
tf.app.flags.DEFINE_string('model_save_path', './model/', 'Directory where to save model weights')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('iteration', 391, 'iteration')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
tf.app.flags.DEFINE_float('epochs', 200, 'epochs')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')

def conv(x, phase, shape):
    he_initializer = tf.contrib.keras.initializers.he_normal()
    W = tf.get_variable('weights', shape=shape, initializer=he_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    # return tf.contrib.layers.batch_norm(x,is_training=phase)    
    return tf.layers.batch_normalization(x,axis=-1,training=phase,name="bn")

def activation(x):
    return tf.nn.relu(x) 

def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def inference(x, phase, keep_prob):
    with tf.variable_scope('conv1'):
        x = conv(x, phase, [5, 5, 3, 192])
        x = activation(x)

    with tf.variable_scope('mlp1-1'):
        x = conv(x, phase, [1, 1, 192, 160])
        x = activation(x)

    with tf.variable_scope('mlp1-2'):
        x = conv(x, phase, [1, 1, 160, 96])
        x = activation(x)

    with tf.name_scope('max_pool-1'):
        x  = max_pool(x, 3, 2)

    with tf.name_scope('dropout-1'):
        x = tf.nn.dropout(x,keep_prob)

    with tf.variable_scope('conv2'):
        x = conv(x, phase, [5, 5, 96, 192])
        x = activation(x)

    with tf.variable_scope('mlp2-1'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp2-2'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.name_scope('max_pool-2'):
        x  = max_pool(x, 3, 2)

    with tf.name_scope('dropout-2'):
        x = tf.nn.dropout(x,keep_prob)

    with tf.variable_scope('conv3'):
        x = conv(x, phase, [3, 3, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp3-1'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp3-2'):
        x = conv(x, phase, [1, 1, 192, 10])
        x = activation(x)

    with tf.name_scope('global_avg_pool'):
        x  = global_avg_pool(x, 8, 1)
        output  = tf.reshape(x,[-1,10])

    return output

def cal_loss(output, y_):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

    with tf.name_scope('l2_loss'):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    tf.summary.scalar("train_loss", cross_entropy)
    return cross_entropy, l2

def training(cost, l2, lr):
    with tf.name_scope('train_op'):
        optimizer = tf.train.MomentumOptimizer(lr, FLAGS.momentum,use_nesterov=True)
        # extra update ops for batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(cost + l2 * FLAGS.weight_decay)
    return train_op

def evaluate(output,y_):
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("train_accuracy", accuracy)
    return accuracy

def lr_schedule(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.002
    return 0.0004

def main(_):
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    # define placeholder x, y_ , keep_prob and learning_rate
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, class_num], name='input_y')
        phase = tf.placeholder(tf.bool, name='phase')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('learning_rate'):
        learning_rate = tf.placeholder(tf.float32)

    # build_network
    output = inference(x,phase,keep_prob)
    loss, l2 = cal_loss(output,y_)
    train_op = training(loss,l2,learning_rate)
    eval_op = evaluate(output,y_)

    summary_op = tf.summary.merge_all()

    # initial an saver to save model
    saver = tf.train.Saver()

    # for testing
    def run_testing(sess, test_x, test_y, loss, eval_op):
        batch_val_loss = []
        batch_val_acc = []
        pre_index = 0
        add = 1000
        for it in range(10):
            test_batch_x = test_x[pre_index:pre_index+add]
            test_batch_y = test_y[pre_index:pre_index+add]
            pre_index = pre_index + add

            loss_, acc_  = sess.run([loss,eval_op],feed_dict={x:test_batch_x, y_:test_batch_y, phase: False, keep_prob: 1.0})

            batch_val_loss.append(loss_)
            batch_val_acc.append(acc_)

        eval_loss, eval_acc = np.mean(batch_val_loss), np.mean(batch_val_acc)
        
        eval_summary = tf.Summary()
        eval_summary.value.add(tag="test_loss", simple_value=eval_loss)
        eval_summary.value.add(tag="test_accuracy", simple_value=eval_acc)

        return eval_acc, eval_loss, eval_summary

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.log_save_path,sess.graph)

        for ep in range(1,FLAGS.epochs+1):
            lr = lr_schedule(ep)
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

                ts = time.time()
                _, batch_loss, sum_op = sess.run([train_op, loss, summary_op],feed_dict={x:batch_x, y_:batch_y, phase:True, keep_prob: FLAGS.dropout, learning_rate: lr})
                te = time.time() - ts

                batch_acc = sess.run(eval_op, feed_dict={x:batch_x, y_:batch_y, phase: False, keep_prob: 1.0})

                pre_index  += FLAGS.batch_size
                train_loss += batch_loss
                train_acc  += batch_acc

                if it == FLAGS.iteration:
                    train_loss /= FLAGS.iteration
                    train_acc /= FLAGS.iteration

                    eval_acc, eval_loss, eval_summary = run_testing(sess, test_x, test_y, loss, eval_op)

                    summary_writer.add_summary(sum_op, ep)
                    summary_writer.add_summary(eval_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, FLAGS.iteration, int(time.time()-start_time), train_loss, train_acc, eval_loss, eval_acc))
                else:
                    print("iteration: %d/%d, cost_time: %.3fs, train_loss: %.4f, train_acc: %.4f" %(it, FLAGS.iteration, te, train_loss / it, train_acc / it) , end='\r')
            if ep % 10 == 0:
                ckpt_path = FLAGS.model_save_path + "model.ckpt%03d"%ep
                save_path = saver.save(sess, ckpt_path)
                print("Model saved in file: %s" % save_path)  

        ckpt_path = FLAGS.model_save_path + "nin.ckpt"
        save_path = saver.save(sess, ckpt_path)
        print("Model saved in file: %s" % save_path)
    
if __name__ == '__main__':
    tf.app.run()

    



          

