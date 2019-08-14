import cv2
import sys
import shutil
import os

VIDEO_PATH = '/data/short_movie/zy_short_movie/video_zy_eye/label_cut/'
SAVE_PATH = './pic_zy/'
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

def make_data():
  subj_list = sorted(os.listdir(VIDEO_PATH))
  f = open(os.path.join(SAVE_PATH, 'list.txt'), 'wb')
  count = 0
  for video in subj_list:
    if video.split('.')[-1] == 'mov'or video.split('.')[-1] == 'mp4':
      emotion = video.split('.')[0].split('_')[0]
      emotion_label = reindex_labels(emotion)
      video_name = os.path.join(VIDEO_PATH, video)
      video_capture = cv2.VideoCapture(video_name)
      font = cv2.FONT_HERSHEY_SIMPLEX
      while True:
        ret, frame = video_capture.read()
        if not ret:
          break
        img_name = emotion+ '_' + str(count) + '.png'
        count += 1
        cv2.imwrite(os.path.join(SAVE_PATH, img_name), frame)
        f.write(img_name + ' '+ str(emotion_label) + '\n')
         
if os.path.exists(SAVE_PATH):
  shutil.rmtree(SAVE_PATH)
os.mkdir(SAVE_PATH)
make_data()
