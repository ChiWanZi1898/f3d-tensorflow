from kinetics_dataset import KineticsDataset
import tensorflow as tf

def get_data(tfrecords_pattern,
             batch_size,
             epoch,
             width,
             height,
             length,
             sample_interval,
             training,
             num_threads,
             num_classes=400):
    dataset = KineticsDataset(
        tfrecords_pattern,
        batch_size,
        epoch,
        width,
        height,
        length,
        sample_interval,
        training,
        num_threads
    )
    frames, labels = dataset.input_fn()
    frames = tf.reshape(frames, (batch_size, length, height, width, 3))
    labels_one_hot = tf.one_hot(labels, num_classes)
    labels_one_hot = tf.reshape(labels_one_hot, (batch_size, num_classes))

    return frames, labels, labels_one_hot
