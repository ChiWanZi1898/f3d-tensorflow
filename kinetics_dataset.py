from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import cv2
import numpy as np
import tensorflow as tf


class KineticsDataset:
    def __init__(
            self,
            rgb_pattern,
            batch_size,
            epoch,
            target_width,
            target_height,
            target_length,
            sample_interval,
            training=True,
            num_threads=None,
            norm=True,
            random_seed=97
    ):
        self.rgb_pattern = rgb_pattern
        self.batch_size = batch_size
        self.epoch = epoch
        self.target_width = target_width
        self.target_height = target_height
        self.target_length = target_length
        self.sample_interval = sample_interval
        self.training = training
        self.random_seed = random_seed
        if num_threads is None:
            self.num_threads = num_threads
        else:
            self.num_threads = self.batch_size

        # convert frames from [0, 255] to [-1, 1]
        self.norm = norm

        self.video_dataset = None
        self.label_dataset = None
        self.dataset = None

        # reset random generator
        np.random.seed(self.random_seed)

    def get_frames(self, video_path):
        video_path = video_path.decode('utf-8')
        # print(video_path)
        # read frames
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for _ in range(length):
            ret, frame = cap.read()
            frames.append(frame)

        # get meta information
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps)
        if len(frames) == 0:
            height = self.target_height
            width = self.target_width
            channel = 3
        else:
            height, width, channel = frames[0].shape
        # print(frames[0].shape)

        # if the length of video is not enough, a few blank frames
        # will be added to its end.
        least_length = self.sample_interval * (self.target_length - 1) + 1
        current_length = length
        length_difference = least_length - current_length
        if length_difference > 0:
            for _ in range(length_difference):
                frames.append(np.zeros((height, width, channel), dtype=np.uint8))

        frames_np = np.asarray(frames, np.uint8)

        # sample
        # If the length of current video has some redundant frames,
        # and if the mode if training, the video will be randomly sampled,
        # otherwise it will just sampled from the beginning.
        first_frame_index = np.random.randint(max(1, -length_difference))
        frames_np = frames_np[
                    first_frame_index:first_frame_index + self.sample_interval * (
                            self.target_length - 1) + 1:self.sample_interval
                    ]

        # if either the current width or height of the frames is
        # smaller than the target ones, the frames will be scaled
        # up with fixed width-to-height ratio.
        ratio_height = 1
        ratio_width = 1
        # if height < self.target_height:
        #     ratio_height = self.target_height / height
        # if width < self.target_width:
        #     ratio_width = self.target_width / width
        # ratio = max(ratio_height, ratio_width)

        h_ratio = self.target_height / height
        w_ratio = self.target_width / width

        ratio = max(h_ratio, w_ratio)

        # print(ratio)
        if ratio != 1:
            height = int(height * ratio)
            width = int(width * ratio)

            # resized_frames = np.zeros([self.target_length, height, width, 3], dtype=np.uint8)
            # for i, frame in enumerate(frames_np):
            #     resized_frame = cv2.resize(frame, (width, height))
            #     resized_frames[i] = resized_frame
            # frames_np = resized_frames

            resized_frames = np.zeros([self.target_length, self.target_height, self.target_width, 3], dtype=np.uint8)
            for i, frame in enumerate(frames_np):
                resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                resized_frames[i] = resized_frame
            frames_np = resized_frames

        cap.release()
        return frames_np

    def parse_video_example(self, serial_example):
        features = {
            'path': tf.FixedLenFeature((), tf.string, default_value=''),
            'name': tf.FixedLenFeature((), tf.string, default_value=''),
            'category': tf.FixedLenFeature((), tf.string, default_value=''),
            'label': tf.FixedLenFeature((), tf.int64, default_value=-1),
        }
        serial_example = tf.reshape(serial_example, [])
        parsed_features = tf.parse_single_example(serial_example, features)

        frames = tf.py_func(self.get_frames, [parsed_features['path']],
                            tf.uint8)

        frames = tf.cast(frames, dtype=tf.float32)
        if self.norm:
            frames = frames / 127.5 - 1
        labels = parsed_features['label']
        return frames, labels

    def get_video_dataset(self, tfrecords_files):
        dataset = tf.data.TFRecordDataset(tfrecords_files)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parse_video_example, batch_size=self.batch_size, num_parallel_calls=self.num_threads))
        # dataset = dataset.batch(self.batch_size)
        self.video_dataset = dataset

    def get_dataset(self):
        dataset = self.video_dataset
        # dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.prefetch(self.batch_size)
        self.dataset = dataset

    def input_fn(self):
        video_tfrecords_files = tf.matching_files(self.rgb_pattern)
        self.get_video_dataset(video_tfrecords_files)
        self.get_dataset()
        iterator = self.dataset.make_one_shot_iterator()

        return iterator.get_next()


def main():
    # just for testing
    rgb_pattern = '/home/zhuhongyu/runspace/kinetics_data/rgb_only/meta/00010.tfrecords'
    # label_pattern = '/mnt/fs1/Dataset/kinetics/train_tfrs_5fps/label_p/data_*.tfrecords'
    kinetics_dataset = KineticsDataset(rgb_pattern, 16, 1, 224, 224, 25, 1)
    x = kinetics_dataset.input_fn()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        t0 = time.time()
        for i in range(200):
            t2 = time.time()
            y = sess.run(x)
            t3 = time.time()
            print(t3 - t2, i, y[0].shape, y[1].shape)
        t1 = time.time()
        print(t1 - t0)


if __name__ == '__main__':
    main()
