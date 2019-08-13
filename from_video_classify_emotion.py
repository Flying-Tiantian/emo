import cv2
import sys
import shutil
import os
import numpy as np
import nltk
from tensorflow.python.platform import gfile
import tensorflow as tf
from sklearn.ensemble import IsolationForest

Sample_rate = 3
VIDEO_PATH = './filmstim/fjh_film_video/'
SAVE_PATH = './result/'

SAVE_KMEANS_PATH = './7_emotions_clip_dataset/features_resnet20_7_pics/'
MODEL = "./resnet20_28_7_new_crop/minimal_graph.proto"

emotion_type = 7

with tf.Session() as persisted_sess:
    print("load graph")
    with gfile.FastGFile(MODEL,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    print("map variables")
    persisted_result = persisted_sess.graph.get_tensor_by_name("Pooling1:0")
    tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)

def reindex_labels(label):
  emotion_dict = { 0 : 'anger' ,
                   1 : 'disgust' ,
                   2 : 'fear' ,
                   3 : 'happiness' ,
                   6 : 'neutral' ,
                   4 : 'sadness' ,
                   5 : 'surprise'
  }
  emotion = emotion_dict[label]
  return emotion

def find_circle_R(kmeans):
    R = []
    centers = []
    for i in range(emotion_type):
        clf = IsolationForest(max_samples=50)
        random_idx = np.random.permutation(range(len(kmeans[i])))
        X_train = kmeans[i][random_idx]
        clf.fit(X_train)
        pred_outliers = clf.predict(np.array(kmeans[i]))
        kmeans_in_circle = kmeans[i][pred_outliers==1]
        center = np.mean(kmeans_in_circle, 0)
        centers.append(center)
        d_point_center = []
        for j in range(len(kmeans_in_circle)):
             d_point_center.append( nltk.cluster.util.cosine_distance(center, kmeans_in_circle[j].tolist()))
        R.append( np.sort(d_point_center)[len(d_point_center)-1] )
    return R, centers


def kmeans_pre(pic_feature, centers, R):
    labels = np.zeros(7)
    distance = []
    for j in range(emotion_type):
        center = centers[j]
        distance.append( nltk.cluster.util.cosine_distance(center, pic_feature)/R[j] )
        
    '''
    distance = np.zeros(emotion_type)
    # more similar, the value of distance more small
    for j in range(emotion_type):
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

def crop_eye_out(I):
  # for happy.mov
  gap = I.shape[1]*7/16
  I = I[I.shape[1]*2/16:I.shape[1]*2/16+gap, I.shape[1]*4/16:I.shape[1]*4/16 + gap]
#  I = I[0:gap, I.shape[1]*4/16:I.shape[1]*4/16 + gap] 
#  gap = I.shape[1]*5/8
#  I = I[0:gap, I.shape[1]*2/8:I.shape[1]*2/8 + gap]
  gray_I = I[:,:,0]
  I[:,:,1] = gray_I
  I[:,:,2] = gray_I
  
  return I

def extract_feature(pic):
    im = cv2.resize(pic.astype('float32'), (28, 28))
    pre = persisted_sess.run(persisted_result, feed_dict={"Placeholder:0":[im]})
    caffe_ft = []
    for num in range(len(pre[0][0][0])):
        caffe_ft.append(pre[0][0][0][num])
    return caffe_ft

def process_single_frame(pic):
    pic = crop_eye_out(pic)
    pic_feature = extract_feature(pic)
    result = kmeans_pre(pic_feature, centers, R)
    return result

def load_kmeans_find_R():
    kmeans = [np.zeros(64)] * 7
    kmeans_file = sorted(os.listdir(SAVE_KMEANS_PATH))
    for file_name in kmeans_file:
        label = file_name.split('.')[0]
        kmean_temp = np.loadtxt(os.path.join(SAVE_KMEANS_PATH, file_name))
        kmeans[int(label)] = kmean_temp
    R, centers = find_circle_R(kmeans)
    return centers, R

if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
os.mkdir(SAVE_PATH)

video_list = sorted(os.listdir(VIDEO_PATH))


centers, R = load_kmeans_find_R()
for video in video_list:
    emotion = video.split('_')[0]
    save_path_one = os.path.join(SAVE_PATH, str(emotion) + '.cvs')
    f = open(save_path_one , 'wb')
    f.write('emotion,weights' + '\n')
    video_capture = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    second = 0
    label_second = -1
    label_temp = np.zeros(emotion_type)
    count = 0
    while True:
        ''' 
        ret, frame = video_capture.read()
        if not ret:
            break
        distances = process_single_frame(frame)
        idx = np.argsort(distances)
        for i in range(3):
            emotion = reindex_labels(idx[i])
            distance = distances[idx[i]]
            f.write(str(emotion) + ' (' + str(distance) + ') ' + ',')
        f.write('\n')
        '''
        ret, frame = video_capture.read()
        if not ret:
            break
        time = video_capture.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        if int(time/1000) == second:
            if  (count % Sample_rate) == 0:
                label = process_single_frame(frame)
                label_temp[label] += 1
            count += 1
        else:
            label_second = np.argmax(label_temp)
            weights = label_temp[label_second]/sum(label_temp)
            second += 1
            label_temp = np.zeros(emotion_type)
            # emotion = reindex_labels(label_second)
            f.write(str(label_second) + ',' + str(weights) + '\n')
            count = 0
        
