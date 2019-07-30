import os
import tensorflow as tf
from tensorflow import graph_util
from models import Inception_model


input_image_shape = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='inputs')
logits = Inception_model(num_classes=7)(input_image_shape)
outputs = tf.identity(logits, 'outputs')
# predicted_classes = tf.argmax(logits, axis=1, name='outputs')


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

graph = tf.get_default_graph()

input_graph_def = graph.as_graph_def()
output_graph_def = graph_util.convert_variables_to_constants(
    sess,
    input_graph_def,
    # We split on comma for convenience
    ["outputs"]
)

# # Finally we serialize and dump the output graph to the filesystem

with tf.gfile.GFile(os.path.join('data', 'Inception_V3_64.pb'), "wb") as f:
    f.write(output_graph_def.SerializeToString())
