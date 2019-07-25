import tensorflow as tf
from tensorflow import graph_util
from models import Mobilenet_64_model


input_image_shape = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='inputs')
logits = Mobilenet_64_model(num_classes=7)(input_image_shape)
outputs = tf.identity(logits, 'outputs')
# predicted_classes = tf.argmax(logits, axis=1, name='outputs')


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state('data/mug_fed_mobilenet64/fresh_train/')

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

graph = tf.get_default_graph()

input_graph_def = graph.as_graph_def()
output_graph_def = graph_util.convert_variables_to_constants(
    sess,
    input_graph_def,
    # We split on comma for convenience
    ["outputs"]
)

# # Finally we serialize and dump the output graph to the filesystem

with tf.gfile.GFile('mobilenet64.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())
