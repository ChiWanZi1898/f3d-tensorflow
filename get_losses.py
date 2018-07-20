import tensorflow as tf


def get_losses(logits,
               labels_one_hot,
               frames,
               reconstructed_frames,
               total_pixel_num,
               in_dict=False):
    lce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.cast(labels_one_hot, tf.float32), logits=logits))
    lflow = tf.nn.l2_loss(0.)
    lrgb = tf.nn.l2_loss(reconstructed_frames[:, ::-1] - frames) / total_pixel_num
    loss = lce + lflow + lrgb

    if not in_dict:
        return loss, lce, lrgb, lflow
    else:
        losses = {}
        losses['loss'] = loss
        losses['lce'] = lce
        losses['lrgb'] = lrgb
        losses['lflow'] = lflow
        return losses
