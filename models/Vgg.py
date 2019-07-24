import tensorflow as tf

class VGG:
    def __init__(self, num_classes, block_sizes, base_filter=64):
        self._num_classes = num_classes
        self._name = 'vgg'
        self._block_sizes = block_sizes
        self._default_input_size = 224
        self._base_filter = base_filter

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def _conv2d_bn(self, x, filters, kernel_size, stride, trainable, is_training, activation=None, name='conv_bn'):
        with tf.variable_scope(name):
            x = tf.layers.Conv2D(filters, kernel_size, strides=(stride, stride), padding='same', activation=activation, trainable=trainable, name='conv').apply(x)
            bn = tf.layers.BatchNormalization(trainable=trainable, name='bn')
            x = bn.apply(x, training=is_training)
            for op in bn.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

        return x

    def _max_pooling(self, x, name='maxpool'):
        return tf.layers.MaxPooling2D(pool_size=2, strides=2, name=name).apply(x)

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            for i, num in enumerate(self._block_sizes):
                for j in range(num):
                    x = self._conv2d_bn(x, 2**i*self._base_filter, 3, 1, trainable, is_training, activation=tf.nn.relu, name='conv_%d_%d' % (i, j))
                x = self._max_pooling(x, name='maxpool_%d' % i)

            x = tf.layers.Conv2D(self._num_classes, 7, strides=(1, 1), padding='valid', activation=None, trainable=trainable, name='final_dense').apply(x)
            x = tf.squeeze(x)
            # x = tf.layers.Dense(units=self._num_classes, trainable=trainable, name='final_dense').apply(x)

        return x


class VGGa_model(VGG):
  def __init__(self, num_classes):
    super().__init__(num_classes, [1, 1, 2, 2, 2])
    self._name = 'vgga'


class VGG16_model(VGG):
  def __init__(self, num_classes):
    super().__init__(num_classes, [2, 2, 3, 3, 3])
    self._name = 'vgg16'


class VGG19_model(VGG):
  def __init__(self, num_classes):
    super().__init__(num_classes, [2, 2, 4, 4, 4])
    self._name = 'vgg19'