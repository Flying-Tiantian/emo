import tensorflow as tf


class LeNet_model:
    def __init__(self, num_classes=10):
        self._num_classes = num_classes
        self._name = 'LeNet'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = tf.layers.Conv2D(6, 5, trainable=trainable, name='conv1').apply(x)
            x = tf.layers.MaxPooling2D(pool_size=2, strides=2, name='maxpool1').apply(x)
            x = tf.layers.Conv2D(16, 5, trainable=trainable, name='conv2').apply(x)
            x = tf.layers.MaxPooling2D(pool_size=2, strides=2, name='maxpool2').apply(x)
            x = tf.layers.Conv2D(120, 5, trainable=trainable, name='fc1').apply(x)
            x = tf.squeeze(x, axis=[1, 2], name='squeeze')
            x = tf.layers.Dense(84, trainable=trainable, name='fc2').apply(x)
            x = tf.layers.Dense(self._num_classes, trainable=trainable, name='fc3').apply(x)

            return x



if __name__ == '__main__':
    image = tf.placeholder(tf.float32, [1, 32, 32, 1])
    ret = LeNet_model()(image)
    pass
