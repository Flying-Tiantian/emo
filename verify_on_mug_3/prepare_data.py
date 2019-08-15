import os
import cv2
import numpy as np
from collections import defaultdict



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


def get_example(image_dir_path, filename_list=None):
    if not filename_list:
        filename_list = sorted(os.listdir(image_dir_path))
    for filename in filename_list:
        if not filename.endswith('.jpg'):
            continue
        image = cv2.imread(os.path.join(image_dir_path, filename))
        resized_image = cv2.resize(image.astype('float32'), (28, 28))
        label = reindex_labels(filename)
        person_name = filename.split('_')[1]

        yield resized_image, label, person_name


def reindex_labels_7(s):
    emotion_dict = {'anger' : 0,
                    'disgust' : 1,
                    'fear' : 2,
                    'happiness' : 3,
                    'neutral' : 6,
                    'sadness' : 4,
                    'surprise': 5}
    for k in emotion_dict:
        if k in s:
            return emotion_dict[k]

    raise ValueError('Input string does not contain any emotions!')

def gen_list(image_dir_path, train=0.8):
    person_filename_dict = defaultdict(lambda: [[] for _ in range(7)])
    for filename in sorted(os.listdir(image_dir_path)):
        if not filename.endswith('.jpg'):
            continue
        person_name = filename.split('_')[1]
        label = reindex_labels_7(filename)
        person_filename_dict[person_name][label].append(filename)

    train_list = []
    test_list = []

    for person_files in person_filename_dict.values():
        for label in range(7):
            total_num = len(person_files[label])
            train_num = int(total_num * train)
            train_list.extend(person_files[label][:train_num])
            test_list.extend(person_files[label][train_num:])

    return train_list, test_list