__author__ = 'H. Zhu'

import numpy as np
import tensorflow as tf

import i3d
from reconstructor import Reconstructor

FLAGS = tf.flags.FLAGS

# IO Settings
tf.flags.DEFINE_string('pretrained_ckpt', 'data/checkpoints/rgb_imagenet/model.ckpt',
                       'Path to checkpoint pretrained on ImageNet.')

# Model Settings
tf.flags.DEFINE_integer('class_num', 400, 'Total number of classes.')
tf.flags.DEFINE_boolean('is_training', True, 'On training mode or not.')
tf.flags.DEFINE_integer('video_length', 32, 'The length of input video (in frame number).')
tf.flags.DEFINE_integer('frame_size', 224, 'The size (both width and height) of video frame.')

# Training settings
tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'The probability to keep in dropout layer.')
tf.flags.DEFINE_integer('batch_size', 1, 'The size of one batch.')


def main(argv):
    inputs = tf.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, FLAGS.video_length, FLAGS.frame_size, FLAGS.frame_size, 3)
    )
    labels = tf.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, 1)
    )

    # I3D
    with tf.variable_scope('RGB'):
        encoder_model = i3d.InceptionI3d(
            FLAGS.class_num, spatial_squeeze=True, final_endpoint='Logits'
        )
        logits, endpoints = encoder_model(
            inputs, is_training=FLAGS.is_training, dropout_keep_prob=FLAGS.dropout_keep_prob
        )
    predictions = tf.nn.softmax(logits)

    # Reconstructor
    with tf.variable_scope('Reconstructor'):
        """Reconstructor
        This reconstructor is used to reconstruct video from 
        high-dimensional features extracted by I3D. It output
        a video that trained to be close to the original video,
        but in a reversed order.
        """
        reconstructor = Reconstructor()
        reconstructed_video = reconstructor.reconstruct(endpoints['Conv3d_2c_3x3'], inputs[:, -1])

    """Loss
    This loss consists 3 parts: lce, lflow and lrgb, 
    lce is the cross entropy loss of video classification, 
    lflow and lrgb are l2 loss for flow stream and
    rgb stream respectively.
    """
    lce = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels, FLAGS.class_num), logits=logits)
    lflow = tf.nn.l2_loss(0)
    lrgb = tf.nn.l2_loss(reconstructed_video[:, ::-1] - inputs)
    loss = lce + lflow + lrgb

    # This is for loading pretrained model.
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    # Optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # Metrics

    # Initializer
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        rgb_saver.restore(sess, FLAGS.pretrained_ckpt)
        rgb_sample = np.load('data/v_CricketShot_g04_c01_rgb.npy')
        feed_dict = {
            inputs: rgb_sample[:, :32]
        }
        ks = [k for k in endpoints.keys()]
        vs = [v for _, v in endpoints.items()]
        vs_r = sess.run(vs, feed_dict=feed_dict)

    for k, v_r in zip(ks, vs_r):
        print(k, v_r.shape)

def train(sess, train_op, metrics):
    pass

if __name__ == '__main__':
    tf.app.run(main)
