from reconstructor import  Reconstructor
from i3d import InceptionI3d
import tensorflow as tf

def build_model(frames, class_num, dropout_keep_prob, reuse, training, ):
    with tf.variable_scope('RGB', reuse=reuse):
        encoder_model = InceptionI3d(
            class_num,
            spatial_squeeze=True,
            final_endpoint='Logits'
        )
        logits, endpoints = encoder_model(
            frames,
            is_training=training,
            dropout_keep_prob=dropout_keep_prob
        )

        predictions = tf.nn.softmax(logits)

    with tf.variable_scope('Reconstructor', reuse=reuse):
        """Reconstructor
                This reconstructor is used to reconstruct video from 
                high-dimensional features extracted by I3D. It output
                a video that trained to be close to the original video,
                but in a reversed order.
        """
        reconstructor = Reconstructor(training=training)
        reconstructed_video = reconstructor.reconstruct(endpoints['Conv3d_2c_3x3'], frames[:, -1])


    return logits, predictions, reconstructed_video