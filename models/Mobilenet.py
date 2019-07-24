import tensorflow as tf

class Mobilenet_model:
    def __init__(self, num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0):
        self._num_classes = num_classes
        self._name = 'MobilenetV1'
        self._default_input_size = round(224 * resolution_multiplier)
        self._width_multiplier = width_multiplier
        self._resolution_multiplier = resolution_multiplier

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def _depthwise_layer(self, x, filters, strides, is_training, trainable):
        x = tf.layers.separable_conv2d(x, filters=x.shape[-1].value, kernel_size=3, strides=strides, padding='same', trainable=trainable)
        x = tf.layers.batch_normalization(x, training=is_training, trainable=trainable, name='bn0')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters=filters, kernel_size=1, strides=1, trainable=trainable)
        x = tf.layers.batch_normalization(x, training=is_training, trainable=trainable, name='bn1')
        x = tf.nn.relu(x)

        return x

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        tf.summary.image('input_images', x, max_outputs=20)
        with tf.variable_scope(self._name, reuse=reuse):
            filters = 32
            x = tf.layers.conv2d(x, filters=int(filters*self._width_multiplier), kernel_size=3, strides=2, padding='same', trainable=trainable, name='conv')

            for unit_num, strides, i in zip([1, 1, 2, 2, 6, 1], [1, 2, 2, 2, 2, 1], range(6)):
                filters = min(filters*2, 1024)
                for unit_index in range(unit_num):
                    with tf.variable_scope('ds_layer_%d_%d' % (i, unit_index)):
                        x = self._depthwise_layer(x, int(filters*self._width_multiplier), strides if unit_index==unit_num-1 else 1, is_training, trainable)

            x = tf.layers.average_pooling2d(x, x.shape[1].value, 1, name='avg_pooling')
            x = tf.layers.dense(x, self._num_classes, name='dense')

        return tf.squeeze(x, axis=[1, 2])


class Mobilenet_160_model(Mobilenet_model):
    def __init__(self, num_classes=10, width_multiplier=1.0, resolution_multiplier=0.714):
        super().__init__(num_classes=num_classes, width_multiplier=width_multiplier, resolution_multiplier=resolution_multiplier)

if __name__ == '__main__':
    
    model = Mobilenet_model(resolution_multiplier=0.714)
    inputs = tf.placeholder(tf.float32, shape=[1, model.get_input_size(), model.get_input_size(), 3])
    logits = model(inputs)
    pass