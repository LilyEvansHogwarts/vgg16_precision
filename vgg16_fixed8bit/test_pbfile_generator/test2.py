from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections

w = 224
h = 224
c = 3


def build_network(height = h, width = w, channel = c):
    x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
    y = tf.placeholder(tf.int64, shape=[None, 2], name='labels')
	
    def conv2d(input, w):
        return tf.nn.conv2d(input, w,[1,1,1,1], padding='SAME')

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    def pool_max(input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME',name=name)


    conv1_1_W = tf.get_variable(name='conv1_1_W',shape=[3,3,3,64],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv1_2_W = tf.get_variable(name='conv1_2_W',shape=[3,3,64,64],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv2_1_W = tf.get_variable(name='conv2_1_W',shape=[3,3,64,128],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv2_2_W = tf.get_variable(name='conv2_2_W',shape=[3,3,128,128],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_1_W = tf.get_variable(name='conv3_1_W',shape=[3,3,128,256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_2_W = tf.get_variable(name='conv3_2_W',shape=[3,3,256,256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_3_W = tf.get_variable(name='conv3_3_W',shape=[3,3,256,256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_1_W = tf.get_variable(name='conv4_1_W',shape=[3,3,256,512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_2_W = tf.get_variable(name='conv4_2_W',shape=[3,3,512,512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_3_W = tf.get_variable(name='conv4_3_W',shape=[3,3,512,512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_1_W = tf.get_variable(name='conv5_1_W',shape=[3,3,512,512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_2_W = tf.get_variable(name='conv5_2_W',shape=[3,3,512,512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_3_W = tf.get_variable(name='conv5_3_W',shape=[3,3,512,512],initializer=tf.zeros_initializer(),dtype=tf.float32)

    conv1_1_b = tf.get_variable(name='conv1_1_b',shape=[64],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv1_2_b = tf.get_variable(name='conv1_2_b',shape=[64],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv2_1_b = tf.get_variable(name='conv2_1_b',shape=[128],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv2_2_b = tf.get_variable(name='conv2_2_b',shape=[128],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_1_b = tf.get_variable(name='conv3_1_b',shape=[256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_2_b = tf.get_variable(name='conv3_2_b',shape=[256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv3_3_b = tf.get_variable(name='conv3_3_b',shape=[256],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_1_b = tf.get_variable(name='conv4_1_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_2_b = tf.get_variable(name='conv4_2_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv4_3_b = tf.get_variable(name='conv4_3_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_1_b = tf.get_variable(name='conv5_1_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_2_b = tf.get_variable(name='conv5_2_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)
    conv5_3_b = tf.get_variable(name='conv5_3_b',shape=[512],initializer=tf.zeros_initializer(),dtype=tf.float32)

    fc6_W = tf.get_variable(name='fc6_W',shape=[25088,4096],initializer=tf.zeros_initializer(),dtype=tf.float32)
    fc7_W = tf.get_variable(name='fc7_W',shape=[4096,4096],initializer=tf.zeros_initializer(),dtype=tf.float32)
    fc8_W = tf.get_variable(name='fc8_W',shape=[4096,1000],initializer=tf.zeros_initializer(),dtype=tf.float32)

    fc6_b = tf.get_variable(name='fc6_b',shape=[4096],initializer=tf.zeros_initializer(),dtype=tf.float32)
    fc7_b = tf.get_variable(name='fc7_b',shape=[4096],initializer=tf.zeros_initializer(),dtype=tf.float32)
    fc8_b = tf.get_variable(name='fc8_b',shape=[1000],initializer=tf.zeros_initializer(),dtype=tf.float32)

    print "conv1_1_W", conv1_1_W.get_shape(), "conv1_1_b", conv1_1_b.get_shape()
    print "conv1_2_W", conv1_2_W.get_shape(), "conv1_2_b", conv1_2_b.get_shape()
    print "conv2_1_W", conv2_1_W.get_shape(), "conv2_1_b", conv2_1_b.get_shape()
    print "conv2_2_W", conv2_2_W.get_shape(), "conv2_2_b", conv2_2_b.get_shape()
    print "conv3_1_W", conv3_1_W.get_shape(), "conv3_1_b", conv3_1_b.get_shape()
    print "conv3_2_W", conv3_2_W.get_shape(), "conv3_2_b", conv3_2_b.get_shape()
    print "conv3_3_W", conv3_3_W.get_shape(), "conv3_3_b", conv3_3_b.get_shape()
    print "conv4_1_W", conv4_1_W.get_shape(), "conv4_1_b", conv4_1_b.get_shape()
    print "conv4_2_W", conv4_2_W.get_shape(), "conv4_2_b", conv4_2_b.get_shape()
    print "conv4_3_W", conv4_3_W.get_shape(), "conv4_3_b", conv4_3_b.get_shape()
    print "conv5_1_W", conv5_1_W.get_shape(), "conv5_1_b", conv5_1_b.get_shape()
    print "conv5_2_W", conv5_2_W.get_shape(), "conv5_2_b", conv5_2_b.get_shape()
    print "conv5_3_W", conv5_3_W.get_shape(), "conv5_3_b", conv5_3_b.get_shape()
    print "fc6_W", fc6_W.get_shape(), "fc6_b", fc6_b.get_shape()
    print "fc7_W", fc7_W.get_shape(), "fc7_b", fc7_b.get_shape()
    print "fc8_W", fc8_W.get_shape(), "fc8_b", fc8_b.get_shape()


# preprocess
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
    images = x - mean 

# conv1
    output_conv1_1 = tf.nn.relu(conv2d(images, conv1_1_W) + conv1_1_b, name='output_conv1_1')
    output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, conv1_2_W) + conv1_2_b, name='output_conv1_2')
    pool1 = pool_max(output_conv1_2, 'pool1')

# conv2
    output_conv2_1 = tf.nn.relu(conv2d(pool1, conv2_1_W) + conv2_1_b, name='output_conv2_1')
    output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, conv2_2_W) + conv2_2_b, name='output_conv2_2')
    pool2 = pool_max(output_conv2_2, 'pool2')

# conv3
    output_conv3_1 = tf.nn.relu(conv2d(pool2, conv3_1_W) + conv3_1_b, name='output_conv3_1')
    output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, conv3_2_W) + conv3_2_b, name='output_conv3_2')
    output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, conv3_3_W) + conv3_3_b, name='output_conv3_3')
    pool3 = pool_max(output_conv3_3,'pool3')

# conv4
    output_conv4_1 = tf.nn.relu(conv2d(pool3, conv4_1_W) + conv4_1_b, name='output_conv4_1')
    output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, conv4_2_W) + conv4_2_b, name='output_conv4_2') 
    output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, conv4_3_W) + conv4_3_b, name='output_conv4_3') 
    pool4 = pool_max(output_conv4_3, 'pool4')

# conv5
    output_conv5_1 = tf.nn.relu(conv2d(pool4, conv5_1_W) + conv5_1_b, name='output_conv5_1')
    output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, conv5_2_W) + conv5_2_b, name='output_conv5_2') 
    output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, conv5_3_W) + conv5_3_b, name='output_conv5_3') 
    pool5 = pool_max(output_conv5_3, 'pool5')

    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1,shape])

# fc6
    output_fc6 = tf.nn.relu(fc(pool5_flat, fc6_W, fc6_b), name='output_fc6')

# fc7
    output_fc7 = tf.nn.relu(fc(output_fc6, fc7_W, fc7_b), name='output_fc7')

# fc8
    output_fc8 = tf.nn.relu(fc(output_fc7, fc8_W, fc8_b), name='output_fc8')

    finaloutput = tf.nn.softmax(output_fc8, name='softmax')

#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y))
#    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    prediction_labels = tf.argmax(finaloutput, axis=1, name='output')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        weights = np.load('vgg16_weights.npz')
        sess.run(tf.assign(conv1_1_W, weights['conv1_1_W']))
        sess.run(tf.assign(conv1_2_W, weights['conv1_2_W']))
        sess.run(tf.assign(conv2_1_W, weights['conv2_1_W']))
        sess.run(tf.assign(conv2_2_W, weights['conv2_2_W']))
        sess.run(tf.assign(conv3_1_W, weights['conv3_1_W']))
        sess.run(tf.assign(conv3_2_W, weights['conv3_2_W']))
        sess.run(tf.assign(conv3_3_W, weights['conv3_3_W']))
        sess.run(tf.assign(conv4_1_W, weights['conv4_1_W']))
        sess.run(tf.assign(conv4_2_W, weights['conv4_2_W']))
        sess.run(tf.assign(conv4_3_W, weights['conv4_3_W']))
        sess.run(tf.assign(conv5_1_W, weights['conv5_1_W']))
        sess.run(tf.assign(conv5_2_W, weights['conv5_2_W']))
        sess.run(tf.assign(conv5_3_W, weights['conv5_3_W']))
        sess.run(tf.assign(fc6_W, weights['fc6_W']))
        sess.run(tf.assign(fc7_W, weights['fc7_W']))
        sess.run(tf.assign(fc8_W, weights['fc8_W']))
        sess.run(tf.assign(conv1_1_b, weights['conv1_1_b']))
        sess.run(tf.assign(conv1_2_b, weights['conv1_2_b']))
        sess.run(tf.assign(conv2_1_b, weights['conv2_1_b']))
        sess.run(tf.assign(conv2_2_b, weights['conv2_2_b']))
        sess.run(tf.assign(conv3_1_b, weights['conv3_1_b']))
        sess.run(tf.assign(conv3_2_b, weights['conv3_2_b']))
        sess.run(tf.assign(conv3_3_b, weights['conv3_3_b']))
        sess.run(tf.assign(conv4_1_b, weights['conv4_1_b']))
        sess.run(tf.assign(conv4_2_b, weights['conv4_2_b']))
        sess.run(tf.assign(conv4_3_b, weights['conv4_3_b']))
        sess.run(tf.assign(conv5_1_b, weights['conv5_1_b']))
        sess.run(tf.assign(conv5_2_b, weights['conv5_2_b']))
        sess.run(tf.assign(conv5_3_b, weights['conv5_3_b']))
        sess.run(tf.assign(fc6_b, weights['fc6_b']))
        sess.run(tf.assign(fc7_b, weights['fc7_b']))
        sess.run(tf.assign(fc8_b, weights['fc8_b']))

        constant_graph = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["output"])
        with tf.gfile.GFile('vggs.pb','wb') as model:
            model.write(constant_graph.SerializeToString())







build_network()
