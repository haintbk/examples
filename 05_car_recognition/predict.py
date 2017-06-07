# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import cifar10


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("checkpoint_dir", "tmp/cifar10_train",
                            """Directory where to read model checkpoints.""")

IMAGE_SIZE = 24
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate_image(filename):
    with tf.Graph().as_default() as g:
        # Number of images to process
        FLAGS.batch_size = 1

        image = img_read(filename)
        logit = cifar10.inference(image)

        output = tf.nn.softmax(logit)
        top_k_pred = tf.nn.top_k(output, k=1)

        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")
                return

            values, indices = sess.run(top_k_pred)
            print indices[0][0], values[0][0]

def img_read(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal("File does not exists %s", filename)
    
    input_img = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    #input_img = tf.image.decode_png(tf.read_file(filename), channels=3)
    reshaped_image = tf.cast(input_img, tf.float32)

    resized_image = tf.image.resize_images(reshaped_image, (IMAGE_SIZE, IMAGE_SIZE))
    float_image = tf.image.per_image_standardization(resized_image)
    image = tf.expand_dims(float_image, 0)  # create a fake batch of images (batch_size = 1)

    return image


def main(argv=None):
    if len(argv) < 2:
        print 'Missing argument : python predict.py [image_path]'
        sys.exit(1)

    filename = argv[1]
    evaluate_image(filename)

if __name__ == '__main__':
  tf.app.run()

