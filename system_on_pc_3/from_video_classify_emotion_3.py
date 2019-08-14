import cv2
import sys
import shutil
import nltk
import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from prepare_data import get_eye_images
from extract_label_features import reindex_labels
from extract_label_features import deal_one_person as generate_one_person_label_features

SAMPLE_RATE = 1

PERSONS = []
FEATURES_DIR = 'features_3'
MODEL = 'minimal_graph.proto'
RESULT_PATH = '~/nemo_short_movie_results_3'

EMOTION_NUM = reindex_labels('')

def get_person_dirs(root_dir):
    if len(PERSONS) == 0:
        return [dirname for dirname in os.listdir(root_dir) if dirname.endswith('_short_movie')]
    else:
        return [dirname for dirname in os.listdir(root_dir) if dirname.endswith('_short_movie') and dirname.split('_')[0] in PERSONS]


def find_circle_R(kmeans):
    R = []
    centers = []
    for i in range(EMOTION_NUM):
        clf = IsolationForest(max_samples=60)
        random_idx = np.random.permutation(range(len(kmeans[i])))
        X_train = kmeans[i][random_idx]
        clf.fit(X_train)
        pred_outliers = clf.predict(np.array(kmeans[i]))
        kmeans_in_circle = kmeans[i][pred_outliers==1]
        center = np.mean(kmeans_in_circle, 0)
        centers.append(center)
        d_point_center = []
        for j in range(len(kmeans_in_circle)):
             d_point_center.append(nltk.cluster.util.cosine_distance(center, kmeans_in_circle[j].tolist()))
        R.append( np.sort(d_point_center)[len(d_point_center)-1] )

    return R, centers


def kmeans_pre(pic_feature, centers, R):
    labels = np.zeros(7)
    distance = []
    for j in range(EMOTION_NUM):
        center = centers[j]
        distance.append( nltk.cluster.util.cosine_distance(center, pic_feature)/R[j] )
        
    '''
    distance = np.zeros(EMOTION_NUM)
    # more similar, the value of distance more small
    for j in range(EMOTION_NUM):
        min_distance = 1
        for k in range(len(kmeans[j])):
            temp_distance = nltk.cluster.util.cosine_distance(pic_feature, kmeans[j][k].tolist())
            if min_distance > temp_distance:
                 min_distance = temp_distance
        distance[j] = min_distance
    '''
    # modify --------------------------------------------------------
    pre_labels = np.argsort(distance)[0]
    #temp = distance/sum(distance)
    #if distance[6]/sum(distance) - temp[0] < 0.01:
    #    return 6    
    #pre_labels = np.argsort(distance)
    return pre_labels


def load_kmeans_find_R(features_dir):
    kmeans = [np.zeros(64)] * EMOTION_NUM
    kmeans_file = sorted(os.listdir(features_dir))
    for file_name in kmeans_file:
        label = file_name.split('.')[0]
        kmean_temp = np.loadtxt(os.path.join(features_dir, file_name))
        kmeans[int(label)] = kmean_temp
    R, centers = find_circle_R(kmeans)
    return centers, R

def frame2emotion(sess, frame, centers, R):
    result_tensor = sess.graph.get_tensor_by_name("Pooling1:0")
    eye_image = cv2.resize(frame.astype('float32'), (28, 28))
    ret = sess.run(result_tensor, feed_dict={"Placeholder:0":[eye_image]})

    feature = ret[0][0][0][:]

    return kmeans_pre(feature, centers, R)


def main(root_dir):
    os.mkdirs(RESULT_PATH, exist_ok=True)
    with tf.Session() as sess:
        print("load graph")
        with open(MODEL,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        # print("map variables")
        # persisted_result = sess.graph.get_tensor_by_name("Pooling1:0")
        # tf.add_to_collection(tf.GraphKeys.VARIABLES, persisted_result)

        person_dirs = get_person_dirs(root_dir)
        for person_dir in person_dirs:
            person_name = person_dir.split('_')[0]

            crop_params = np.fromfile(os.path.join(root_dir, person_dir, 'crop_param.npy'), dtype=float, sep='|')
            crop_params = tuple(crop_params)

            features_dir = os.path.join(RESULT_PATH, person_dir, 'features_3')
            os.mkdirs(features_dir, exist_ok=True)

            generate_one_person_label_features(sess, os.path.join(root_dir, person_dir), features_dir)
            centers, R = load_kmeans_find_R(features_dir)

            video_dir_path = os.path.join(root_dir, person_dir, 'video_'+person_name+'_eye', 'test')
            for video_name in os.listdir(video_dir_path):
                if not ('.mp4' in video_name or '.mov' in video_name):
                    continue
                video_path = os.path.join(video_dir_path, video_name)

                emotion = video_name.split('.')[0]
                save_path = os.path.join(RESULT_PATH, person_dir, 'result', str(emotion) + '.cvs')
                with open(save_path , 'w') as f:
                    f.write('emotion,weights' + '\n')
                    second = 0
                    label_second = -1
                    label_temp = np.zeros(EMOTION_NUM)
                    count = 0
                    for frame, time in get_eye_images(video_path, crop_params, get_time=True):
                        if  (count % SAMPLE_RATE) == 0:
                            if int(time/1000) == second:
                                label = frame2emotion(sess, frame, centers, R)
                                label_temp[label] += 1
                            else:
                                label_second = np.argmax(label_temp)
                                weights = label_temp[label_second]/sum(label_temp)
                                second += 1
                                label_temp = np.zeros(EMOTION_NUM)
                                f.write(str(label_second) + ',' + str(weights) + '\n')

                        count += 1



if __name__ == '__main__':
    main(sys.argv[1])
