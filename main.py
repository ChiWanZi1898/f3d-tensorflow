__author__ = 'H. Zhu'

import os

import tensorflow as tf

from build_model import build_model
from configs import get_configs
from get_data import get_data
from get_losses import get_losses


def main(configs):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Dataset
    frames_train, labels_train, labels_one_hot_train = get_data(
        configs.train_tfrecords_pattern,
        configs.batch_size,
        configs.epoch,
        configs.frame_size,
        configs.frame_size,
        configs.video_length,
        configs.sample_interval,
        configs.training,
        configs.num_threads,
        configs.num_classes
    )

    frames_test, labels_test, labels_one_hot_test = get_data(
        configs.test_tfrecords_pattern,
        configs.batch_size,
        configs.epoch,
        configs.frame_size,
        configs.frame_size,
        configs.video_length,
        configs.sample_interval,
        configs.training,
        configs.num_threads,
        configs.num_classes
    )

    # I3D
    logits_train, predictions_train, reconstructed_video_train = build_model(
        frames_train,
        configs.num_classes,
        configs.dropout_keep_prob,
        reuse=False,
        training=configs.training
    )

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    logits_test, predictions_test, reconstructed_video_test = build_model(
        frames_test,
        configs.num_classes,
        configs.dropout_keep_prob,
        reuse=True,
        training=False
    )

    loss_train, lce_train, lrgb_train, lflow_train = get_losses(
        logits_train,
        labels_one_hot_train,
        frames_train,
        reconstructed_video_train,
        total_pixel_num=configs.batch_size * configs.frame_size * configs.frame_size * 3 * configs.video_length)

    loss_test, lce_test, lrgb_test, lflow_test = get_losses(
        logits_test,
        labels_one_hot_test,
        frames_test,
        reconstructed_video_test,
        total_pixel_num=configs.batch_size * configs.frame_size * configs.frame_size * 3 * configs.video_length)

    # Metrics
    top_5_train = tf.nn.top_k(predictions_train, 5)
    top_5_test = tf.nn.top_k(predictions_test, 5)

    # Optimizer
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=configs.learning_rate)
    train_op = optimizer.minimize(loss_train, global_step=global_step)

    # Initializer
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        if configs.from_imagenet:
            restore_from_imagenet(sess, configs.imagenet_pretrained, rgb_variable_map)
        if configs.from_saved:
            restore(sess, configs.tfrecords_load_dir)

        def test_func():
            test(sess, labels_test, top_5_test, logits_test, lce_test, lrgb_test, lflow_test, configs.max_test_num)

        train(sess, train_op, labels_train, top_5_train, loss_train, lce_train, lrgb_train, lflow_train, global_step,
              configs.max_train_num,
              configs.test_interval, configs.save_interval, configs.tfrecords_save_dir, test_func)


def train(sess,
          train_op,
          labels,
          top5,
          loss,
          lce,
          lrgb,
          lflow,
          global_step,
          max_train_num,
          test_interval,
          save_interval,
          save_path,
          test_func):
    print('Start training...')
    for i in range(max_train_num):
        try:
            _, e_labels, e_top5, e_loss, e_lce, e_lflow, e_lrgb, e_global_step = sess.run(
                [train_op, labels, top5, loss, lce, lflow, lrgb, global_step])
            tf.logging.info(
                'TRAIN {}steps: loss - {}, lce - {}, lflow - {}, lrgb - {}'.format(e_global_step, e_loss, e_lce,
                                                                                   e_lflow, e_lrgb))
            tf.logging.debug(
                'labels: {}, top5s: {}'.format(e_labels, e_top5)
            )
        except tf.errors.OutOfRangeError:
            print('End of dataset.')
            break

        if e_global_step % test_interval == 0:
            test_func()

        if e_global_step % save_interval == 0:
            save(sess, save_path, global_step)


def test(sess, labels, top5, loss, lce, lrgb, lflow, max_test_num):
    total_counter = 0
    top1_counter = 0
    top5_counter = 0
    for i in range(max_test_num):
        try:
            e_labels, e_top5, e_loss, e_lce, e_lflow, e_lrgb, = sess.run([labels, top5, loss, lce, lrgb, lflow])
            top1_counter_batch = 0
            top5_counter_batch = 0
            total_counter_batch = 0
            for l, t5 in zip(e_labels, e_top5):
                total_counter += 1
                total_counter_batch += 1
                if l in t5:
                    top5_counter += 1
                    top5_counter_batch += 1
                elif l == t5[0]:
                    top1_counter += 1
                    top1_counter_batch += 1
            tf.logging.info(
                'TEST {}steps: top1 - {}, top5 - {}, top1_batch - {}, top5_batch - {},loss - {}, lce - {}, lflow - {}, lrgb - {}'.format(
                    i, top1_counter / total_counter, top5_counter / total_counter,
                       top1_counter_batch / total_counter_batch, top5_counter_batch / total_counter_batch, e_loss,
                    e_lce,
                    e_lflow, e_lrgb))
            tf.logging.debug(
                'labels: {}, top5s: {}'.format(e_labels, e_top5)
            )

        except tf.errors.OutOfRangeError:
            print('End of dataset.')
            break


def save(sess, save_path, global_step):
    saver = tf.train.Saver()
    saver.save(sess, save_path, global_step=global_step)


def restore(sess, restore_path):
    saver = tf.train.Saver()
    saver.restore(sess, restore_path)


def restore_from_imagenet(sess, restore_path, variable_map):

    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver.restore(sess, restore_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    configs = get_configs()
    main(configs)
