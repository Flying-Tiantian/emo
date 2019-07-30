import tensorflow as tf


class Inception_model:
    def __init__(self, num_classes=7):
        self._num_classes = num_classes
        self._name = 'Inception_V3'
        self._model = tf.keras.applications.InceptionV3(include_top=True,
                                                        weights=None,
                                                        classes=num_classes)

    def __call__(self, x, trainable=True, is_training=True, reuse=False):
        logits = self._model(inputs=x, training=is_training)

        return logits
