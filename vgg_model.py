import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from scipy import misc

tf.logging.set_verbosity(tf.logging.INFO)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

with tf.device('/cpu:0'):
    def vgg_model(X, y_, reuse, is_train):

        W_init = tf.truncated_normal_initializer(stddev=5e-2)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
      
            net = InputLayer(X, name='input')

            # Convolutional Layer #1
            net = Conv2d(
                net,
                n_filter=64,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv1')
            #128 128 64

            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')

            # Convolutional Layer #2
            net = Conv2d(
                net,
                n_filter=64,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv2')
            #128 128 64

            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch2')
            # Pooling Layer #1
            #pooling layer with a 2x2 filter and stride of 2
            net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME',name='pool1')
            #64 64 64

            # Convolutional Layer #3
            net = Conv2d(
                net,
                n_filter=128,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv3')
            # 64 64 128
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch3')


            # # Convolutional Layer #4
            net = Conv2d(
                net,
                n_filter=128,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv4')
        
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch4')
            #64 64 128

            # Pooling Layer #2
            net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME',name='pool2')
            
            # 32 32 128
            # Convolutional Layer #5
            net = Conv2d(
                net,
                n_filter=256,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv5')
        
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch5')
            # 32 32 256
            # Convolutional Layer #6
            net = Conv2d(
                net,
                n_filter=256,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv6')
        
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch6')
            #32 32 256
            # Convolutional Layer #7
            net = Conv2d(
                net,
                n_filter=256,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv7')
        
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch7')
            #32 32 256

            # Pooling Layer #3
            net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME',name='pool3')
        
            #16 16 256
            # Convolutional Layer #8
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv8')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch8')
            # 16 16 512
            # Convolutional Layer #9
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv9')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch9')
            #16 16 512
            # Convolutional Layer #10
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv10')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch10')
            #16 16 512

            # Pooling Layer #4
            net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME',name='pool4')
            #8 8 512

            # Convolutional Layer #11
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv11')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch11')
            # 8 8 512
            # Convolutional Layer #12
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv12')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch12')
            #8 8 512
            # Convolutional Layer #13
            net = Conv2d(
                net,
                n_filter=512,
                filter_size=(3, 3),
                strides=(1,1),
                padding='SAME',
                W_init=W_init, 
                b_init=None, 
                name='conv13')
            net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch13')
            #8 8 512

            # Pooling Layer #5
            net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME',name='pool5')
            #4 4 512

            # Flatten tensor into a batch of vectors
            net = FlattenLayer(net, name='flatten')

            # Dense Layer
            net = DenseLayer(net, n_units=4096, act=tf.nn.relu,
                            W_init=W_init2, b_init=b_init2, name='dense1')     
            net = DenseLayer(net, n_units=4096, act=tf.nn.relu,
                            W_init=W_init2, b_init=b_init2, name='dense2')  
            net = DenseLayer(net, n_units=1000, act=tf.nn.relu,
                            W_init=W_init2, b_init=b_init2, name='dense3')  
            net = DenseLayer(net, n_units=2, act = tf.identity,
                            W_init=tf.truncated_normal_initializer(stddev=1/192.0),
                            name='output') 
            y = net.outputs
            ce = tl.cost.cross_entropy(y, y_, name='cost')
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            L2 = tf.contrib.layers.l2_regularizer(0.004)(net.all_params[4]) + \
                    tf.contrib.layers.l2_regularizer(0.004)(net.all_params[6])
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, acc


def main(argv):

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    learning_rate = 0.0001
    batch_size = 120
    model_file_name = "./models/model_vgg_tumor.ckpt"
    resume = False
    n_epoch = 1
    n_step_epoch = int(float(3460000) / batch_size)
    n_step = n_epoch * n_step_epoch
    path = './Tfrecord/'
    eval_step = 300
    eval_times = 10
    print_freq = 10

    filenames = os.listdir(path)
    for (i,f) in enumerate(filenames):
        filenames[i] = path + f

    print filenames


    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=n_epoch)
        image, label = read_and_decode(filename_queue, 128, 128, 4)
        image_batch, label_batch = tf.train.batch(
          [image, label], batch_size=batch_size)

        with tf.device('/gpu:0'): # <-- remove it if you don't have GPU
            # you may want to try batch normalization
            network, cost, acc, = vgg_model(image_batch, label_batch, None, is_train=True)
            _, cost_test, acc_test = vgg_model(image_batch, label_batch, True, is_train=False)
            train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                epsilon=1e-08, use_locking=False).minimize(cost)

        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tl.layers.initialize_global_variables(sess)
        if resume:
            print("Load existing model " + "!"*10)
            saver = tf.train.Saver()
            saver.restore(sess, model_file_name)

        network.print_params(False)
        network.print_layers()

        print('   learning_rate: %f' % learning_rate)
        print('   batch_size: %d' % batch_size)
        print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))
        shouldEval = False

        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss, train_acc, n_batch = 0, 0, 0
            
            for s in range(n_step_epoch):
                if s % eval_step == 0 and s != 0:
                    shouldEval = True
                    test_loss, test_acc, n_test_batch = 0, 0, 0

                if shouldEval:
                    err, ac = sess.run([cost_test, acc_test])
                    test_loss += err
                    test_acc += ac
                    n_test_batch += 1
                
                    print("Testing Epoch %d : Step %d of %d took %fs" % (epoch, s, n_step_epoch, time.time() - start_time))
                    print("   test loss: %f" % (test_loss/ n_test_batch))
                    print("   test acc: %f" % (test_acc/ n_test_batch))

                    if n_test_batch >= eval_times:
                        shouldEval = False

                else:
                    err, ac, _ = sess.run([cost, acc, train_op])
                    
                    train_loss += err
                    train_acc += ac
                    n_batch += 1
                    if s % print_freq == 0:
                        print("Epoch %d : Step %d of %d took %fs" % (epoch, s, n_step_epoch, time.time() - start_time))
                        print("   train loss: %f" % (train_loss/ n_batch))
                        print("   train acc: %f" % (train_acc/ n_batch))

        
            print("Save model " + "!"*10)
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def read_and_decode(queue, W, H, C):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    # get feature from serialized example
    # decode
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [W, H, C])
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int32)

    return img, label

if __name__ == "__main__":
    tf.app.run()

