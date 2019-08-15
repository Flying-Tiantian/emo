import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def one_hot_embedding(label, n_classes):
    """
    One-hot embedding
    Args:
        label: int32 tensor [B]
        n_classes: int32, number of classes
    Return:
        embedding: tensor [B x n_classes]
    """
    embedding_params = np.eye(n_classes, dtype=np.float32)
    with tf.device('/cpu:0'):
        params = tf.constant(embedding_params)
        embedding = tf.gather(params, label)
    return embedding

def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    with tf.variable_scope(scope):
        # kernel = tf.Variable(
        #   tf.truncated_normal([k, k, n_in, n_out],
        #     stddev=math.sqrt(2/(k*k*n_in))),
        #   name='weight')
        kernel = tf.get_variable('weight', [k, k, n_in, n_out], initializer = tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
        if bias:
            bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('biases', bias)
            conv = tf.nn.bias_add(conv, bias)
    return conv

def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    Args:
        x: Tensor, 4D BHWD input maps
        n_out: integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope: string, variable scope
        affine: whether to affine-transform outputs
    Return:
        normed: batch-normalized maps
    """
    with tf.variable_scope(scope):
        # beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
        #   name='beta', trainable=True)
        # gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
        #   name='gamma', trainable=affine)
        beta = tf.get_variable('beta', [n_out], initializer = tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', [n_out], initializer = tf.constant_initializer(1.0))
        tf.add_to_collection('biases', beta)
        tf.add_to_collection('weights', gamma)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        # def mean_var_with_update():
        #   ema_apply_op = ema.apply([batch_mean, batch_var])
        #   with tf.control_dependencies([ema_apply_oe]):
        #     return tf.identity(batch_mean), tf.identity(batch_var)
        # mean, var = tf.cond(phase_train,
        #   mean_var_with_update,
        #   lambda: (ema.average(batch_mean), ema.average(batch_var)))
        mean, var = batch_mean, batch_var
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, 
            beta, gamma, 1e-3, affine)
    return normed

def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
            shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME',
                                False, scope='shortcut')
        else:
            y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
            shortcut = tf.identity(x, name='shortcut')
        y = batch_norm(y, n_out, phase_train, scope='bn_1')
        y = tf.nn.relu(y, name='relu_1')
        y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
        y = batch_norm(y, n_out, phase_train, scope='bn_2')
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y

def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
        for i in range(n - 1):
            y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
    return y


def residual_net_shared(x, n, n_classes, phase_train, scope='shared_net'):
    with tf.variable_scope(scope):
        y = conv2d(x, 3, 16, 3, 1, 'SAME', False, scope='conv_init')
        y = batch_norm(y, 16, phase_train, scope='bn_init')
        y = tf.nn.relu(y, name='relu_init')
        y = residual_group(y, 16, 16, n, False, phase_train, scope='group_1')
        return y

def residual_net_rest(y, n, n_classes, phase_train, scope = 'rest_net'):
    with tf.variable_scope(scope):
        y = residual_group(y, 16, 32, n, True, phase_train, scope='group_2')
        y = residual_group(y, 32, 64, n, True, phase_train, scope='group_3')
        y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
        y = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last')
        #y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
        y = tf.squeeze(y, squeeze_dims=[1, 2])
    return y

def siamese_net(y, phase_train, scope='sia_net'):
    with tf.variable_scope(scope):
        y = conv2d(y, 16, 32, 3, 2, 'SAME', False, scope='conv_init_1')
        y = conv2d(y, 32, 64, 3, 2, 'SAME', False, scope='conv_init_2')
        y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
        y = tf.squeeze(y, squeeze_dims=[1, 2])
    return y


class emo_model:
    def __init__(self, num_classes=3):
        self._num_classes = num_classes
        self._name = 'emo_net'
        self._default_input_size = 32

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._default_input_size

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            x = residual_net_shared(x, 2, self._num_classes, is_training, scope='shared_net')
            x = residual_net_rest(x, 6, self._num_classes, is_training, scope = 'rest_net')
            
            return x
