import tensorflow as tf

FLAGS = tf.flags.FLAGS

# IO Settings
tf.flags.DEFINE_string('imagenet_pretrained', 'data/checkpoints/rgb_imagenet/model.ckpt',
                       'Path to checkpoint pretrained on ImageNet.')
tf.flags.DEFINE_string('train_tfrecords_pattern', '/home/zhuhongyu/runspace/kinetics_data/rgb_path_only/meta/*.tfrecords',
                       'Pattern of tfrecords file.')
tf.flags.DEFINE_string('test_tfrecords_pattern', '/home/zhuhongyu/runspace/kinetics_data/rgb_path_only_test/meta/*.tfrecords',
                       'Pattern of tfrecords file.')
tf.flags.DEFINE_integer('num_threads', 16, 'The number of threads for reading data.')
tf.flags.DEFINE_boolean('from_imagenet', True, 'Whether train from imagenet pretrained model')
tf.flags.DEFINE_boolean('from_saved', False, 'Whether train from saved model')
tf.flags.DEFINE_string('tfrecords_load_dir', '', 'The directory to load tfrecords files')
tf.flags.DEFINE_string('tfrecords_save_dir', '', 'The directory to save tfrecords files.')

# Model Settings
tf.flags.DEFINE_integer('num_classes', 400, 'Total number of classes.')
tf.flags.DEFINE_boolean('training', True, 'On training mode or not.')
tf.flags.DEFINE_integer('video_length', 32, 'The length of input video (in frame number).')
tf.flags.DEFINE_integer('frame_size', 224, 'The size (both width and height) of video frame.')


# Training settings
tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'The probability to keep in dropout layer.')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
tf.flags.DEFINE_integer('batch_size', 8, 'The size of one batch.')
tf.flags.DEFINE_integer('max_train_num', 100000, 'The maximum number of testing iteration.')
tf.flags.DEFINE_integer('save_interval', 5000, 'The interval of saving the model.')

# Testing settings
tf.flags.DEFINE_integer('test_interval', 2500, 'The interval of testing the model')
tf.flags.DEFINE_integer('max_test_num', 100, 'The maximum number of testing iteration.')

# Dataset settings
tf.flags.DEFINE_integer('epoch', 10, 'The number of epoch.')
tf.flags.DEFINE_integer('sample_interval', 1, 'The sample rate, sample one in how many frames.')


def get_configs():
    return FLAGS
