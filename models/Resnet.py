import tensorflow as tf


class Resnet_model:
    def __init__(self, 
                 resnet_size,
                 use_bottleneck,
                 num_classes,
                 num_filters,
                 kernel_size,
                 conv_stride,
                 first_pool_size,
                 first_pool_stride,
                 block_sizes,
                 block_strides):
        self._resnet_size = resnet_size
        self._use_bottleneck = use_bottleneck
        self._num_classes = num_classes
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._conv_stride = conv_stride
        self._first_pool_size = first_pool_size
        self._first_pool_stride = first_pool_stride
        self._block_sizes = block_sizes
        self._block_strides = block_strides

    def _conv2d_bn(self, x, filters, kernel_size, stride, trainable, is_training, activation=None, name='conv_bn'):
        with tf.variable_scope(name):
            x = tf.layers.Conv2D(filters, kernel_size, strides=(stride, stride), padding='same', activation=activation, trainable=trainable, name='conv').apply(x)
            bn = tf.layers.BatchNormalization(trainable=trainable, name='bn')
            x = bn.apply(x, training=is_training)
            for op in bn.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

        return x

    def _residual_block(self, x, filters, stride, trainable, is_training):
        in_channel = x.shape.dims[-1].value

        if filters == in_channel and stride == 1:
            shortcut = x
        else:
            shortcut = self._conv2d_bn(x, filters, 1, stride, trainable, is_training, name='conv_shortcut')
        x = self._conv2d_bn(x, filters, 3, stride, trainable, is_training, activation=tf.nn.relu, name='conv_0')
        x = self._conv2d_bn(x, filters, 3, 1, trainable, is_training, name='conv_1')
        x = tf.nn.relu(x + shortcut)

        return x

    def _bottleneck(self, x, filters, stride, trainable, is_training):
        in_channel = x.shape.dims[-1].value

        if filters * 4 == in_channel:
            shortcut = x
        else:
            shortcut = self._conv2d_bn(x, filters * 4, 1, stride, trainable, is_training, name='conv_shortcut')
        x = self._conv2d_bn(x, filters, 1, stride, trainable, is_training, activation=tf.nn.relu, name='conv_0')
        x = self._conv2d_bn(x, filters, 3, 1, trainable, is_training, activation=tf.nn.relu, name='conv_1')
        x = self._conv2d_bn(x, filters * 4, 1, 1, trainable, is_training, name='conv_2')
        x = tf.nn.relu(x + shortcut)

        return x

    def _build_layer(self, x, use_bottleneck, block_num, filters, stride, trainable, is_training):
        if self._use_bottleneck:
            building_block = self._bottleneck
            name = 'bottleneck_'
        else:
            building_block = self._residual_block
            name = 'residual_block_'

        for i in range(block_num):
            with tf.variable_scope(name + str(i)):
                x = building_block(x, filters, stride if i==0 else 1, trainable, is_training)

        return x


    def __call__(self, x, trainable=True, is_training=True):
        x = self._conv2d_bn(x, self._num_filters, self._kernel_size, self._conv_stride, trainable, is_training, activation=tf.nn.relu, name='conv1')
        x = tf.layers.MaxPooling2D(pool_size=self._first_pool_size, strides=self._first_pool_stride, trainable=trainable, name='maxpool1').apply(x)
        for i, size in enumerate(self._block_sizes):
            filters = 2**i * self._num_filters
            with tf.variable_scope('layer' + str(i)):
                x = self._build_layer(x, self._use_bottleneck, size, filters, self._block_strides[i], trainable, is_training)

        map_size = x.shape.dims[1].value
        x = tf.layers.AveragePooling2D(pool_size=map_size, strides=map_size, trainable=trainable, name='avgpool1').apply(x)
        x = tf.squeeze(x, axis=[1, 2], name='squeeze')
        x = tf.layers.Dense(units=self._num_classes, trainable=trainable, name='final_dense').apply(x)

        return x


class Resnet18_model(Resnet_model):
    def __init__(self, num_classes):
        super().__init__(
            resnet_size=20,
            use_bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=[2, 2, 2, 2],
            block_strides=[2, 2, 2, 2])
        self._name = 'resnet18'
        self._default_input_size = 224

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = super().__call__(x, trainable=trainable, is_training=is_training)

        return x


class Resnet20_model(Resnet_model):
    def __init__(self, num_classes):
        super().__init__(
            resnet_size=20,
            use_bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=1,
            first_pool_stride=1,
            block_sizes=[3, 3, 3],
            block_strides=[1, 2, 2])
        self._name = 'resnet20'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = super().__call__(x, trainable=trainable, is_training=is_training)

        return x


class Resnet20_128_model(Resnet_model):
    def __init__(self, num_classes):
        super().__init__(
            resnet_size=20,
            use_bottleneck=False,
            num_classes=num_classes,
            num_filters=128,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=1,
            first_pool_stride=1,
            block_sizes=[3, 3, 3],
            block_strides=[1, 2, 2])
        self._name = 'resnet20_128'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = super().__call__(x, trainable=trainable, is_training=is_training)

        return x


class Resnet56_model(Resnet_model):
    def __init__(self, num_classes):
        super().__init__(
            resnet_size=56,
            use_bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=1,
            first_pool_stride=1,
            block_sizes=[9, 9, 9],
            block_strides=[1, 2, 2])
        self._name = 'resnet56'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = super().__call__(x, trainable=trainable, is_training=is_training)

        return x


class Resnet56_64_model(Resnet_model):
    def __init__(self, num_classes):
        super().__init__(
            resnet_size=56,
            use_bottleneck=False,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=1,
            first_pool_stride=1,
            block_sizes=[9, 9, 9],
            block_strides=[1, 2, 2])
        self._name = 'resnet56_64'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = super().__call__(x, trainable=trainable, is_training=is_training)

        return x



if __name__ == '__main__':
    model = Resnet56_model(100)
    x = tf.placeholder(tf.float32, shape=(2, 32, 32, 3))
    x = model(x)
    pass