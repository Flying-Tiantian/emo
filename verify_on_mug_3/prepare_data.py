import os
import cv2
import numpy as np



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


def get_example(image_dir_path):
    for filename in os.listdir(image_dir_path):
        if not filename.endswith('.jpg'):
            continue
        image = cv2.imread(os.path.join(image_dir_path, filename))
        resized_image = cv2.resize(image.astype('float32'), (28, 28))
        label = reindex_labels(filename)
        person_name = filename.split('_')[1]

        yield resized_image, label, person_name
