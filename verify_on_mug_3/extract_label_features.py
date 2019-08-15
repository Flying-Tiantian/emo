import os
import sys
import cv2
import shutil
import numpy as np
import tensorflow as tf
from collections import defaultdict
from prepare_data import get_example, reindex_labels


PERSONS = []
CLASS_NUM = reindex_labels('')

def filter_person(person, persons):
    if len(persons) == 0:
        return True
    return person in persons

def extract_label_features(sess, image_dir_path, persons, file_list=None):
    result_tensor = sess.graph.get_tensor_by_name("Pooling1:0")
    person_features_dict = defaultdict(lambda: [[] for _ in range(CLASS_NUM)])

    print('Extract label features from dir %s...' % image_dir_path)
    count = 0
    for eye_image, label, person_name in get_example(image_dir_path, file_list):
        ret = sess.run(result_tensor, feed_dict={"Placeholder:0":[eye_image]})
        feature = ret[0][0][0][:]
        person_features_dict[person_name][label].append(feature)

        count += 1
        print('\rDone: %d' % count, end='')

    print('')

    return person_features_dict


def save_features(result_dir, person_features_dict):
    for person_name in person_features_dict:
        features_dir = os.path.join(result_dir, person_name, 'features_3')
        if os.path.exists(features_dir):
            shutil.rmtree(features_dir)
        os.makedirs(features_dir)
        for label, features in enumerate(person_features_dict[person_name]):
            np.savetxt(os.path.join(features_dir, str(label)+'.txt'), features)



def main(image_dir_path, sess=None, persons=PERSONS, result_dir='/home/tian/nemo_mug_results_3', file_list=None):
    if sess is None:
        with tf.Session() as sess:
            person_features_dict = extract_label_features(sess, image_dir_path, persons, file_list)
            save_features(result_dir, person_features_dict)
    else:
        person_features_dict = extract_label_features(sess, image_dir_path, persons, file_list)
        save_features(result_dir, person_features_dict)



if __name__ == '__main__':
    main(sys.argv[1])
