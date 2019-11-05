# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:53:45 2019

@author: sb00747428
"""

import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())
default_device = "/gpu:0"



train_data = numpy.load('train_data.npy')
train_labels= numpy.load('train_labels.npy')



''' Building a model for a GAN '''

BASE_LEARNING_RATE = 0.0002
BATCH_SIZE=40
RANDOM_INPUT_DIMENSIONALITY = 100
MAX_EPOCH=100
INCLUDE_NOISE=True
LOGDIR="./mnist_gan_logs/lr_{}_include_noise_{}_batchsize_{}".format(BASE_LEARNING_RATE, INCLUDE_NOISE, BATCH_SIZE)
RESTORE=False
TRAINING=True

tf.reset_default_graph()
g = tf.Graph()

with tf.device(default_device):
    with g.as_default():
        # Input noise to the generator:
        noise_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, RANDOM_INPUT_DIMENSIONALITY], name="noise")
#         fake_input   = tf.reshape(noise_tensor, (tf.shape(noise_tensor)[0], 10,10, 1))

        # Placeholder for the discriminator input:
        real_flat  = tf.placeholder(tf.float32, [BATCH_SIZE, 40, 32, 3], name='x')

        # We augment the input to the discriminator with gaussian noise
        # This makes it harder for the discriminator to do it's job, preventing
        # it from always "winning" the GAN min/max contest
        real_noise = tf.placeholder(tf.float32, [BATCH_SIZE, 40, 32, 3], name="real_noise")
        fake_noise = tf.placeholder(tf.float32, [BATCH_SIZE, 40, 32, 3], name="fake_noise")

        real_images = real_flat + real_noise


def build_discriminator(input_tensor, reuse, is_training, reg=0.2, dropout_rate=0.3):
    # Use scoping to keep the variables nicely organized in the graph.
    # Scoping is good practice always, but it's *essential* here as we'll see later on
    with tf.variable_scope("mnist_discriminator", reuse=reuse):

        x = tf.layers.dense(input_tensor, 512, name="fc1")

        # Apply a non linearity:
        x = tf.maximum(reg*x, x, name="leaky_relu_1")

        # Apply a dropout layer:
        x = tf.layers.dropout(x,rate=dropout_rate, training=is_training, name="dropout1")

        x = tf.layers.dense(x, 256, name="fc2")

        # Apply a non linearity:
        x = tf.maximum(reg*x, x, name="leaky_relu_2")

        # Apply a dropout layer:
        x = tf.layers.dropout(x,rate=dropout_rate, training=is_training, name="dropout2")

        x = tf.layers.dense(x, 1, name="fc4")

        # Since we want to predict "real" or "fake", an output of 0 or 1 is desired.  sigmoid is perfect for this:
        x = tf.nn.sigmoid(x, name="discriminator_sigmoid")

        return x

with tf.device(default_device):
    with g.as_default():
        real_image_logits = build_discriminator(real_images, reuse=False,is_training=TRAINING, reg=0.2, dropout_rate=0.3)


def build_generator(input_tensor, reg=0.2):
    # Again, scoping is essential here:
    with tf.variable_scope("mnist_generator"):
        x = tf.layers.dense(input_tensor, 256, name="fc1")

        # Apply a non linearity:
        x = tf.maximum(reg*x, x, name="leaky_relu_1")

        x = tf.layers.dense(x, 512, name="fc2")

        # Apply a non linearity:
        x = tf.maximum(reg*x, x, name="leaky_relu_2")

        x = tf.layers.dense(x, 1024, name="fc3")

        # Apply a non linearity:
        x = tf.maximum(reg*x, x, name="leaky_relu_3")

        x = tf.layers.dense(x, 40*32*3, name="fc4")


        # Reshape to match mnist images:
        x = tf.reshape(x, (-1, 40, 32, 3))

        # The final non linearity applied here is to map the images onto the [-1,1] range.
        x = tf.nn.tanh(x, name="generator_tanh")
        return x


with tf.device(default_device):
    with g.as_default():
        fake_images = build_generator(noise_tensor) + fake_noise


with tf.device(default_device):
    with g.as_default():
        fake_image_logits = build_discriminator(fake_images, reuse=True, is_training=TRAINING, dropout_rate=0.3, reg=0.2)



        '''Loss function'''

with tf.device(default_device):
    # Build the loss functions:
    with g.as_default():
        with tf.name_scope("cross_entropy") as scope:

            # Be careful with the loss functions.  The sigmoid activation is already applied as the
            # last step of the discriminator network above.  If you want to use something like
            # tf.nn.sigmoid_cross_entropy_with_loss, it *will not train* because it applies
            # a sigmoid a second time.
            d_loss_total = -tf.reduce_mean(tf.log(real_image_logits) + tf.log(1. - fake_image_logits))


            # This is the adverserial step: g_loss tries to optimize fake_logits to one,
            # While d_loss_fake tries to optimize fake_logits to zero.
            g_loss = -tf.reduce_mean(tf.log(fake_image_logits))

            # This code is useful if you'll use tensorboard to monitor training:
#             d_loss_summary = tf.summary.scalar("Discriminator_Real_Loss", d_loss_real)
#             d_loss_summary = tf.summary.scalar("Discriminator_Fake_Loss", d_loss_fake)
            d_loss_summary = tf.summary.scalar("Discriminator_Total_Loss", d_loss_total)
            d_loss_summary = tf.summary.scalar("Generator_Loss", g_loss)

with tf.device(default_device):
    with g.as_default():
        with tf.name_scope("accuracy") as scope:
            # Compute the discriminator accuracy on real data, fake data, and total:
            accuracy_real  = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_image_logits),
                                                             tf.ones_like(real_image_logits)),
                                                    tf.float32))
            accuracy_fake  = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_image_logits),
                                                             tf.zeros_like(fake_image_logits)),
                                                    tf.float32))

            total_accuracy = 0.5*(accuracy_fake +  accuracy_real)

            # Again, useful for tensorboard:
            acc_real_summary = tf.summary.scalar("Real_Accuracy", accuracy_real)
            acc_real_summary = tf.summary.scalar("Fake_Accuracy", accuracy_fake)
            acc_real_summary = tf.summary.scalar("Total_Accuracy", total_accuracy)

with tf.device(default_device):
    with g.as_default():
        with tf.name_scope("training") as scope:
            # Global steps are useful for restoring training:
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            # Make sure the optimizers are only operating on their own variables:

            all_variables      = tf.trainable_variables()
            discriminator_vars = [v for v in all_variables if v.name.startswith('mnist_discriminator/')]
            generator_vars     = [v for v in all_variables if v.name.startswith('mnist_generator/')]


            discriminator_optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE, 0.5).minimize(
                d_loss_total, global_step=global_step, var_list=discriminator_vars)
            generator_optimizer     = tf.train.AdamOptimizer(BASE_LEARNING_RATE, 0.5).minimize(
                g_loss, global_step=global_step, var_list=generator_vars)


with tf.device(default_device):
   with g.as_default():
        # Reshape images for snapshotting:
    fake_images_reshaped = tf.reshape(fake_images, (-1, 40, 32, 3))
    real_images_reshaped = tf.reshape(real_images, (-1, 40, 32, 3))
    tf.summary.image('fake_images', fake_images_reshaped, max_outputs=4)
    tf.summary.image('real_images', real_images_reshaped, max_outputs=4)


'''Training the networks'''

with tf.device(default_device):
    with g.as_default():
        merged_summary = tf.summary.merge_all()

        # Set up a saver:
        train_writer = tf.summary.FileWriter(LOGDIR)




epochs   = [] # store the epoch corresponding to the variables below
gen_loss = []
dis_loss = []
images   = []
true_acc = []
fake_acc = []
tot_acc  = []


with tf.device(default_device):
    with g.as_default():
        sess = tf.InteractiveSession()
        if not RESTORE:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            saver = tf.train.Saver()
        else:
            latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
            print ("Restoring model from {}".format(latest_checkpoint))
            saver = tf.train.Saver()
            saver.restore(sess, latest_checkpoint)



        print ("Begin training ...")
        # Run training loop
        y = train_data.shape[0]
        for i in range(50000000):
            for bla in range (0,y,BATCH_SIZE):
                real_data = train_data[bla:bla+40,:,:,:]
                step = sess.run(global_step)

                # Receive data (this will hang if IO thread is still running = this
                # will wait for thread to finish & receive data)
                epoch = (1.0*i*BATCH_SIZE) / 4400.
                if (epoch > MAX_EPOCH):
                    break
                sigma = max(0.75*(10. - epoch) / (10), 0.05)

                # Update the generator:
                # Prepare the input to the networks:
                fake_input = numpy.random.normal(loc=0, scale=1, size=(BATCH_SIZE, RANDOM_INPUT_DIMENSIONALITY))



#                real_data, label = mnist.train.next_batch(BATCH_SIZE)
#                real_data = 2*(real_data - 0.5)
                if INCLUDE_NOISE:
                    real_noise_addition = numpy.random.normal(scale=sigma,size=(BATCH_SIZE,40,32,3))
                    fake_noise_addition = numpy.random.normal(scale=sigma,size=(BATCH_SIZE,40,32,3))
                else:
                    real_noise_addition = numpy.zeros((BATCH_SIZE, 40,32,3))
                    fake_noise_addition = numpy.zeros((BATCH_SIZE, 40,32,3))

                # Update the discriminator:
                [generated_mnist, _] = sess.run([fake_images,
                                                discriminator_optimizer],
                                                feed_dict = {noise_tensor : fake_input,
                                                             real_flat : real_data,
                                                             real_noise: real_noise_addition,
                                                             fake_noise: fake_noise_addition})

                # Update the generator:
                fake_input = numpy.random.normal(loc=0, scale=1, size=(BATCH_SIZE, RANDOM_INPUT_DIMENSIONALITY))
                if INCLUDE_NOISE:
                    fake_noise_addition = numpy.random.normal(scale=sigma,size=(BATCH_SIZE,40,32,3))
                else:
                    fake_noise_addition = numpy.zeros((BATCH_SIZE, 40,32,3))


                [ _ ] = sess.run([generator_optimizer],
                    feed_dict = {noise_tensor: fake_input,
                                 real_flat : real_data,
                                 real_noise: real_noise_addition,
                                 fake_noise: fake_noise_addition})

                # Run a summary step:
                [summary, g_l, d_l, acc_fake, acc_real, acc] = sess.run(
                    [merged_summary, g_loss, d_loss_total, accuracy_fake, accuracy_real, total_accuracy],
                    feed_dict = {noise_tensor : fake_input,
                                 real_flat : real_data,
                                 real_noise: real_noise_addition,
                                 fake_noise: fake_noise_addition})


                train_writer.add_summary(summary, step)


                if step != 0 and step % 500 == 0:
                    saver.save(
                        sess,
                        LOGDIR+"/checkpoints/save",
                        global_step=step)


                # train_writer.add_summary(summary, i)
                # sys.stdout.write('Training in progress @ step %d\n' % (step))
                if i != 0 and int(10*epoch) == 10*epoch:
                    if int(epoch) == epoch:
                        print ('Training in progress @ epoch %g, g_loss %g, d_loss %g accuracy %g' % (epoch, g_l, d_l, acc))
                    epochs.append(epoch)
                    gen_loss.append(g_l)
                    dis_loss.append(d_l)
                    images.append(generated_mnist)
                    true_acc.append(acc_real)
                    fake_acc.append(acc_fake)
                    tot_acc.append(acc)
