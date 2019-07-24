import os
import tensorflow as tf
from absl import flags
from absl import app

from models import *
from inputs.dataset_gen import dataset_generator



clas_models = {
    'mobilenet': Mobilenet_model,
}


def compute_decay_steps(batch_size, train_example_num, decay_epochs):
    return int(train_example_num / batch_size * decay_epochs)


def make_model_fn(model_class, decay_steps, model_dir):
    def model_fn(features, labels, mode):
        logits = model_class(features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

        probabilities = tf.nn.softmax(logits)
        predicted_classes = tf.argmax(logits, axis=1)

        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes, name='acc_op')
        accuracy_top_5 = tf.metrics.mean(
            tf.nn.in_top_k(predictions=probabilities, targets=labels, k=5, name='top_5_op'))

        metrics = {
            'accuracy': accuracy,
            'accuracy_top_5': accuracy_top_5
        }


        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes,
                'probabilities': probabilities,
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        else:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                logits=logits, labels=labels)

            tf.identity(cross_entropy, name='cross_entropy')
            tf.summary.scalar('cross_entropy', cross_entropy)

            global_step = tf.train.get_or_create_global_step()

            lr = tf.train.exponential_decay(flags.FLAGS.lr, global_step, decay_steps=decay_steps, decay_rate=0.1, staircase=True)
            tf.identity(lr, name='learning_rate')
            tf.summary.scalar('learning_rate', lr)

            def exclude_batch_norm(name):
                return ('batch_normalization' not in name) and ('bn' not in name)
            l2_loss = flags.FLAGS.weight_decay * tf.add_n(
                # loss is computed using fp32 for numerical stability.
                [
                    tf.nn.l2_loss(tf.cast(v, tf.float32))
                    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    if exclude_batch_norm(v.name)
                ])

            tf.summary.scalar('l2_loss', l2_loss)

            loss = cross_entropy + l2_loss
            tf.identity(loss, name='total_loss')
            tf.summary.scalar('total_loss', loss)

            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

                tf.identity(accuracy[1], name='train_accuracy')
                tf.summary.scalar('train_accuracy', accuracy[1])

                train_op = tf.group([train_op] + tf.get_collection(tf.GraphKeys.UPDATE_OPS))  # UPDATE_OPS is used for batch_norm layers!!!

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=2**31, # only save at the first step
                                output_dir= os.path.join(model_dir, "eval"),
                                summary_op=tf.summary.merge_all())


                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[eval_summary_hook])

    return model_fn


def train_and_eval():
    dataset_gen = dataset_generator(flags.FLAGS.task)
    num_classes = dataset_gen.get_class_num()

    model_dir = os.path.join(flags.FLAGS.model_dir, flags.FLAGS.task + '_' + flags.FLAGS.target_model)

    target_model = clas_models[flags.FLAGS.target_model](num_classes=num_classes)

    model_dir = os.path.join(model_dir, 'fresh_train')
    target_model_trainable = True

    model_class = target_model

    train_example_num, _ = dataset_gen.get_image_num()
    decay_steps = compute_decay_steps(flags.FLAGS.batch_size, train_example_num, flags.FLAGS.decay_epochs)

    train_hooks = []
    train_hooks.append(tf.train.LoggingTensorHook(
        tensors={
            'learning_rate': 'learning_rate',
            'train_accuracy': 'train_accuracy',
            'cross_entropy': 'cross_entropy',
            'total_loss': 'total_loss'},
        every_n_iter=100))

    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60*60*24,
        save_checkpoints_steps=None,
        keep_checkpoint_max=1,
        session_config=tf_config)
    classifier = tf.estimator.Estimator(
        model_fn=make_model_fn(model_class, decay_steps, model_dir),
        model_dir=model_dir,
        config=run_config
    )

    epoch = 0

    if flags.FLAGS.eval_only:
        eval_results = classifier.evaluate(input_fn=lambda: dataset_gen.input_fn(False, target_model.get_input_size(), flags.FLAGS.batch_size, 1))
        print('[Evaluate]epoch: final, accuracy: %f' % eval_results['accuracy'])
        return

    for i in range(flags.FLAGS.train_epochs//flags.FLAGS.epochs_between_evals):
        classifier.train(
            input_fn=lambda: dataset_gen.input_fn(True, target_model.get_input_size(), flags.FLAGS.batch_size, flags.FLAGS.epochs_between_evals),
            hooks=train_hooks)
        eval_results = classifier.evaluate(input_fn=lambda: dataset_gen.input_fn(False, target_model.get_input_size(), flags.FLAGS.batch_size, 1))
        epoch += flags.FLAGS.epochs_between_evals
        print('[Evaluate]epoch: %d, accuracy: %f' % (epoch, eval_results['accuracy']))

    with open(os.path.join(flags.FLAGS.model_dir, flags.FLAGS.task + '_' + flags.FLAGS.target_model, 'results.txt'), 'a') as f:
        f.write(model_dir + (" acc: %f\n" % eval_results['accuracy']))


def define_flags():
    flags.DEFINE_enum(
        name='task', default='mug_fed',
        enum_values=['mug_fed', 'mug_fed_crop'],
        help='Which dataset to use, and what task to do.')
    flags.DEFINE_enum(
        name='target_model', default='mobilenet',
        enum_values=['mobilenet'],
        help='Which target model to use.')
    flags.DEFINE_boolean(
        name='eval_only', default=False,
        help='Skip training and only perform evaluation on the latest checkpoint.')
    flags.DEFINE_float(
        name='weight_decay', short_name='wd', default=1e-4,
        help='Weight decay coefficient for l2 regularization.')
    flags.DEFINE_float(
        name='learning_rate', short_name='lr', default=0.001, lower_bound=0.0,
        help='Initial learning rate.')
    flags.DEFINE_integer('batch_size', default=128, lower_bound=0,
                         help='Batch size.')
    flags.DEFINE_integer('train_epochs', default=500, lower_bound=0,
                         help='The epoch num for train.')
    flags.DEFINE_integer('decay_epochs', default=50000, lower_bound=0,
                         help='Epoch number between lr decay.')
    flags.DEFINE_integer('epochs_between_evals', default=5, lower_bound=0,
                         help='Eval between how many epochs.')

    flags.DEFINE_string(
        name='model_dir', default='data',
        help='The root data dir.')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval()

def debug_flags():
    flags.FLAGS.task = 'cifar100_coarse'
    flags.FLAGS.target_model = 'resnet56'
    flags.FLAGS.mode = 'train'
    flags.FLAGS.wd = 0.0
    flags.FLAGS.train_epochs = 150
    flags.FLAGS.transform_model = 'unet_2_2_16_0.0'
    flags.FLAGS.invert_model = 'unet_2_2_16_0.0'

if __name__ == '__main__':
    define_flags()
    # debug_flags()
    app.run(main)
