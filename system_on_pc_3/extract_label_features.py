import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from prepare_data import get_eye_images


PERSONS = []


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


def extract_label_features(sess, video_path, crop_params, save_path):
    result_tensor = sess.graph.get_tensor_by_name("Pooling1:0")
    features = []
    for eye_image in get_eye_images(video_path, crop_params):
        eye_image = cv2.resize(eye_image.astype('float32'), (28, 28))
        ret = sess.run(result_tensor, feed_dict={"Placeholder:0":[eye_image]})
        features.append(ret[0][0][0][:])

    np.savetxt(save_path, features)


def get_person_dirs(root_dir, persons):
    if len(persons) == 0:
        return [dirname for dirname in os.listdir(root_dir) if dirname.endswith('_short_movie')]
    else:
        return [dirname for dirname in os.listdir(root_dir) if dirname.endswith('_short_movie') and dirname.split('_')[0] in persons]


def deal_one_person(sess, person_dir, features_dir):
    person_name = os.path.split(person_dir)[-1].split('_')[0]
    video_dir_path = os.path.join(person_dir, 'video_'+person_name+'_eye', 'label_cut')
    crop_params = np.fromfile(os.path.join(person_dir, 'crop_param.npy'), dtype=float, sep='|')
    crop_params = tuple(crop_params)
    for video_name in os.listdir(video_dir_path):
        if not ('.mp4' in video_name or '.mov' in video_name):
            continue
        video_path = os.path.join(video_dir_path, video_name)
        extract_label_features(sess, video_path, crop_params, os.path.join(features_dir, str(reindex_labels(video_name))+'.txt'))


def main(root_dir, sess=None, persons=PERSONS, features_dir='features_3'):
    person_dirs = get_person_dirs(root_dir, persons, features_dir)

    if sess is None:
        with tf.Session() as sess:
            print("load graph")
            with open("minimal_graph.proto",'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            for person_dir in person_dirs:
                deal_one_person(sess, os.path.join(root_dir, person_dir), os.path.join(root_dir, person_dir, features_dir))
    else:
        for person_dir in person_dirs:
            deal_one_person(sess, os.path.join(root_dir, person_dir), os.path.join(root_dir, person_dir, features_dir))



if __name__ == '__main__':
    main(sys.argv[1])
