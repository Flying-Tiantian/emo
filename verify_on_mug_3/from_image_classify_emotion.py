import cv2
import sys
import shutil
import nltk
import os
import pprint
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from prepare_data import reindex_labels, get_example
from extract_label_features import main as extract_label_features



PERSONS = []
FEATURES_DIR = 'features_3'
MODEL = 'minimal_graph.proto'
RESULT_PATH = '/home/tian/nemo_mug_results_3'

EMOTION_NUM = reindex_labels('')



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
        train_data_dir = os.path.join(root_dir, 'train_crop_eye_new')
        test_data_dir = os.path.join(root_dir, 'test_crop_eye_new')

        # extract_label_features(train_data_dir, sess, PERSONS, RESULT_PATH)

        center_R_dict = {}
        for person_name in os.listdir(RESULT_PATH):
            try:
                center_R_dict[person_name] = load_kmeans_find_R(os.path.join(RESULT_PATH, person_name, FEATURES_DIR))
            except:
                pass

        result_file_path = os.path.join(RESULT_PATH, 'pre_result.csv')

        true_list = []
        pred_list = []
        count = 0
        print('Predict from dir %s...' % test_data_dir)
        with open(result_file_path, 'w') as f:
            f.write('label, prediction\n')
            result_tensor = sess.graph.get_tensor_by_name("Pooling1:0")
            for eye_image, label, person_name in get_example(test_data_dir):
                if not person_name in center_R_dict:
                    continue
                prediction = frame2emotion(sess, eye_image, *center_R_dict[person_name])
                true_list.append(label)
                pred_list.append(prediction)
                f.write('%d, &%d\n' % (label, prediction))

                count += 1
                print('\rDone: %d' % count, end='')

        print('')

        pprint(classification_report(true_list, pred_list, target_names=['positive', 'negtive', 'neutral']))




if __name__ == '__main__':
    main(sys.argv[1])
