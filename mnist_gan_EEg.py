# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:52:24 2019

@author: sb00747428
"""

import numpy as np
import tensorflow as tf
#from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt

def build_generator(gan_input):
    with tf.compat.v1.variable_scope('generator'):
        x = tf.compat.v1.layers.dense(gan_input, 256, activation='relu')
        x = tf.compat.v1.layers.dense(x, 512, activation='relu')
        # x = tf.compat.v1.layers.dropout(x, 0.1)
        x = tf.compat.v1.layers.dense(x, 1024, activation='relu')
        # x = tf.compat.v1.layers.dropout(x, 0.1)
        x = tf.compat.v1.layers.dense(x, 2048, activation='relu')
        x = tf.compat.v1.layers.dense(x, 40*32*3, activation='tanh')

    return x


def build_discriminator(disc_input, reuse=False):
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
        x = tf.compat.v1.layers.dense(disc_input, 1024, activation='relu')
        x = tf.compat.v1.layers.dense(x, 512, activation='relu')
        x = tf.compat.v1.layers.dense(x, 64, activation='relu')
        x = tf.compat.v1.layers.dense(x, 1)

    return x


def discriminator_loss(cross_entropy, real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2

    return total_loss


def wgan_discriminator_loss(real_output, fake_output):
    total_loss = tf.compat.v1.reduce_mean(fake_output) - tf.compat.v1.reduce_mean(real_output)

    return total_loss


def begin():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    gan_input_dim = 100

    # load data
    #(train_images, train_labels), (_, _) = load_data()
    train_images1 = np.load('train_data.npy')
    train_labels1= np.load('train_labels.npy')
    xxx= np.where(train_labels1 == 1)

    for Value in xxx:
        xxx2 = train_images1[Value,:,:,:]

    train_images = xxx2
    train_labels= train_labels1
    train_images = (train_images - 0.5) / 0.5

    # create input placeholders for generator and discriminator
    gen_input_plc = tf.compat.v1.placeholder(tf.float32, [None, gan_input_dim])
    disc_input_plc = tf.compat.v1.placeholder(tf.float32, [None, 3840])

    # create the generator and discriminator
    gen_output = build_generator(gen_input_plc)
    disc_real_output = build_discriminator(disc_input_plc)
    disc_fake_output = build_discriminator(gen_output, True)

    # create the generator and discriminator losses
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # disc_loss = discriminator_loss(cross_entropy, disc_real_output, disc_fake_output)
    disc_loss = wgan_discriminator_loss(disc_real_output, disc_fake_output)
    # gen_loss = cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)
    gen_loss = -tf.reduce_mean(disc_fake_output)

    # create optimizers for generator and discriminator
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    disc_step = tf.compat.v1.train.RMSPropOptimizer(0.00001).minimize(disc_loss, var_list=disc_vars)
    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    gen_step = tf.compat.v1.train.RMSPropOptimizer(0.00001).minimize(gen_loss, var_list=gen_vars)
    # op for clipping discriminator weights
    clip_disc_w = [tf.compat.v1.assign(p, tf.compat.v1.clip_by_value(p, -0.01, 0.01)) for p in disc_vars]

    n_epoch = 100
    batch_sz = 11
    n_disc_update = 10
    n_batch = train_images.shape[0] // batch_sz
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        #saver.restore(sess, 'model/mnist.ckpt')
        sess.run(tf.compat.v1.global_variables_initializer())

        for e in range(n_epoch):
            epoch_dloss = 0
            epoch_gloss = 0

            for b in range(n_batch):
                x = np.reshape(train_images[(b * batch_sz):((b + 1) * batch_sz), :, :], (batch_sz, -1))

                gen_noise_input = np.random.normal(size=[batch_sz, gan_input_dim])
                for d in range(n_disc_update):
                    ins_noise_real = np.random.normal(loc=0, scale=0.1, size=x.shape)
                    _, d_loss = sess.run([disc_step, disc_loss], feed_dict={disc_input_plc: x,
                                                                            gen_input_plc: gen_noise_input})
                    sess.run(clip_disc_w)
                _, g_loss, g_output = sess.run([gen_step, gen_loss, gen_output], feed_dict={gen_input_plc: gen_noise_input})


                epoch_dloss += d_loss
                epoch_gloss += g_loss

            print('Epoch: %d; G-loss: %f; D-loss: %f' % (e, g_loss / n_batch, d_loss / n_batch))
            sample_id =  np.random.randint(batch_sz)
            xxy = g_output[sample_id,:]
            x2 = xxy.reshape(40,32,3)
            plt.imshow(x2)
            plt.show()

        saver.save(sess, 'model/mnist.ckpt')

    print('Done...')


if __name__ == '__main__':
    begin()
