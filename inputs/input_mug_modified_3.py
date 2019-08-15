import os
import tensorflow as tf
from .image_preprocessing import preprocess_image


PROPERTIES = {
    'num_classes': 3,
    'num_images': {
        'train': 5055,
        'val': 859,
    }
}

def reindex_labels(s):
    if s == '':
        return 3
        
    emotion_dict = {'anger' : 1,
                    'disgust' : 1,
                    'fear' : 1,
                    'happiness' : 0,
                    'neutral' : 2,
                    'sadness' : 1,
                    'surprise': 0}
    for k in emotion_dict:
        if k in s:
            return emotion_dict[k]

    raise ValueError('Input string does not contain any emotions!')

DATA_DIR = os.path.join('data', 'mug_fed_data')

def creat_list(data_dir, is_training):
    dirname = 'train' if is_training else 'test'
    dirpath = os.path.join(data_dir, dirname+'_crop_eye_new')

    paths = []
    labels = []
    for filename in os.listdir(dirpath):
        if not filename.endswith('.jpg'):
            continue
        paths.append(os.path.join(dirpath, filename))
        labels.append(reindex_labels(filename))

    return (paths, labels)


def input_fn(is_training,
             image_size,
             batch_size,
             num_epochs,
             data_dir=DATA_DIR,
             dtype=tf.float32):
    dataset = tf.data.Dataset.from_tensor_slices(creat_list(data_dir, is_training))

    def _parse_function(path, label):
        raw = tf.read_file(path)
        return preprocess_image(raw, image_size, image_size, is_training=is_training), label

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    dataset = input_fn(False, 224, 16, 1)
    pass
