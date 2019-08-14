import os
import cv2
import numpy as np
from scipy import spatial
import nltk
from sklearn import decomposition
import shutil
from tensorflow.python.platform import gfile
import tensorflow as tf
import os

pic_path_data = 'pic_crop_zy'

emotion_type = 7
SAVE_FEATURE_PATH = './features_resnet20_7_pics_zy/'

def reindex_labels(emotion):
  emotion_dict = { 'anger' : 0,
                   'disgust' : 1,
                   'fear' : 2,
                   'happiness' : 3,
                   'neutral' : 6,
                   'sadness' : 4,
                   'surprise': 5
  }
  label = emotion_dict[emotion]
  return label

def extract_feature_labels(pic_path):
  with tf.Session() as persisted_sess:
    print("load graph")
    with gfile.FastGFile("./minimal_graph.proto",'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      persisted_sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
    print("map variables")
    persisted_result = persisted_sess.graph.get_tensor_by_name("Pooling1:0")
    tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
  
  temp_set = []
  temp_set_labels = []
  pics_list = sorted(os.listdir(pic_path))
  for pic in pics_list:
    label = reindex_labels(pic.split('_')[0])
    im = cv2.imread(os.path.join(pic_path, pic))
    im = cv2.resize(im.astype('float32'), (28, 28))
    pre = persisted_sess.run(persisted_result, feed_dict={"Placeholder:0":[im]})

    caffe_ft = []
    for num in range(len(pre[0][0][0])):
      caffe_ft.append(pre[0][0][0][num])


    temp_set.append(caffe_ft)
    temp_set_labels.append(label)
  return temp_set, temp_set_labels


data_set, data_set_labels = extract_feature_labels(pic_path_data)
if os.path.exists(SAVE_FEATURE_PATH):
  shutil.rmtree(SAVE_FEATURE_PATH)
os.mkdir(SAVE_FEATURE_PATH)


for i in range(emotion_type):
  idx = np.array(data_set_labels) == i 
  data = np.array(data_set)[idx]
  np.savetxt(os.path.join(SAVE_FEATURE_PATH, str(i) +'.txt'), data)


