# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:30:01 2019

@author: sb00747428
"""

import numpy as np # linear algebra
import scipy.io
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting



n_classes = 2

def load_data():
    traindata = scipy.io.loadmat('Traindata.mat', squeeze_me=True)
    trainlabels = scipy.io.loadmat('trainlabels.mat', squeeze_me=True)
    Xvalidation = scipy.io.loadmat('Xvalidation.mat', squeeze_me=True)
    Yvalidation = scipy.io.loadmat('Yvalidation.mat', squeeze_me=True)

    traindata =  traindata['traindata']
    trainlabels =  trainlabels['labels']
    Xvalidation = Xvalidation['XValidation']
    Yvalidation = Yvalidation['YValidation']


    trainingdata = []
    traininglabels = []
    Xval = []
    Yval =[]

    for i in range(100):
        td = traindata[i]
        trainingdata.append(td)
        tl = trainlabels[i]
        traininglabels.append(tl)

    for j in range(10):
        X= Xvalidation[j]
        Xval.append(X)
        Y= Yvalidation[j]
        Yval.append(Y)

    return trainingdata, traininglabels, Xval, Yval



with tf.name_scope('inputs'):
    x_input = tf.placeholder(tf.float32, shape=[None, 22, 182, 199, 3])
    y_input = tf.placeholder(tf.float32, shape=[None, n_classes])


def cnn_model(x_train_data, keep_rate=0.7, seed=None):

    with tf.name_scope("layer_a"):
        # conv => 16*16*16
        conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 16*16*16
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 8*8*8
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)

    with tf.name_scope("layer_c"):
        # conv => 8*8*8
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 8*8*8
        conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 4*4*4
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)

    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 4*4*4*128])
        dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
        # (1-keep_rate) is the probability that the node will be kept
        dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=True)

    with tf.name_scope("y_conv"):
        y_conv = tf.layers.dense(inputs=dropout, units=4)

    return y_conv

def train_neural_network(x_train_data, y_train_data, x_test_data, y_test_data, learning_rate=0.05, keep_rate=0.7, epochs=10, batch_size=128):


    with tf.name_scope("cross_entropy"):
        prediction = cnn_model(x_input, keep_rate, seed=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))

    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    iterations = int(len(x_train_data)/batch_size) + 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = datetime.datetime.now()

        iterations = int(len(x_train_data)/batch_size) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0
            # mini batch
            for itr in range(iterations):
                mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
                epoch_loss += _cost

            #  using mini batch in case not enough memory
            acc = 0
            itrs = int(len(x_test_data)/batch_size) + 1
            for itr in range(itrs):
                mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})

            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:',acc/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

if __name__ == "__main__":
   trainingdata, traininglabels, Xval, Yval =  load_data()
   x_train = np.asarray(trainingdata)
   x_train = x_train.astype(dtype = 'float64')
   x_test = np.asarray(Xval)
   x_test = x_test.astype(dtype = 'float64')
   y_train = np.asarray(traininglabels)
   y_test = np.asarray(Yval)

   cnn_model(x_train)
