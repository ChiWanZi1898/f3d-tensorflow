__author__ = 'H. Zhu'

import tensorflow as tf


class Reconstructor:
    def __init__(self, training=True):
        self.training = training

    def _decode(self, inputs):
        """decode
        This
        :param inputs:
        :return:
        """
        net = inputs
        net = self._deconv2d(net, 512)
        net = self._deconv2d(net, 256)
        net = self._deconv2d(net, 128)
        net = self._deconv2d(net, 3, activation=tf.tanh, batch_norm=False)
        return net

    def _encode(self, inputs):
        net = inputs
        net = self._conv2d(net, 128, batch_norm=False)
        net = self._conv2d(net, 256)
        net = self._conv2d(net, 512)
        net = self._conv2d(net, 1024)
        return net

    def _get_background(self, inputs):
        return self._decode(self._encode(inputs))

    def _deconv2d(self, inputs, out, kernel=4, stride=2, activation=tf.nn.relu, batch_norm=True, e=1e-3):
        outputs = tf.layers.conv2d_transpose(
            inputs, out, kernel, stride, padding='same'
        )
        if batch_norm:
            outputs = tf.layers.batch_normalization(outputs, epsilon=e, training=self.training)
        if activation is not None:
            outputs = activation(outputs)
        return outputs

    def _deconv3d(self, inputs, out, kernel=(4, 8, 8), stride=(2, 4, 4), activation=tf.nn.relu, batch_norm=True,
                  e=1e-3):
        outputs = tf.layers.conv3d_transpose(
            inputs, out, kernel, stride, padding='same'
        )
        if batch_norm:
            outputs = tf.layers.batch_normalization(outputs, epsilon=e, training=self.training)
        if activation is not None:
            outputs = activation(outputs)
        return outputs

    def _conv2d(self, inputs, out, kernel=4, stride=2, activation=tf.nn.relu, batch_norm=True, e=1e-3):
        outputs = tf.layers.conv2d(
            inputs, out, kernel, stride, padding='same'
        )
        if batch_norm:
            outputs = tf.layers.batch_normalization(outputs, epsilon=e, training=self.training)
        if activation is not None:
            outputs = activation(outputs)
        return outputs

    def reconstruct(self, inputs, first_frame):
        background = self._get_background(first_frame)
        background_expand = tf.expand_dims(background, 1)
        background_expand = tf.tile(background_expand, [1, 32, 1, 1, 1])
        foreground = self._deconv3d(inputs, 3)
        mask = self._deconv3d(inputs, 1)
        outputs = mask * foreground + (1 - mask) * background_expand
        return outputs
