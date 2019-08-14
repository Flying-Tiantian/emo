import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

pre_labels_path = './result_wh'
emotion_type = 7
pre_file_list =sorted(os.listdir(pre_labels_path))
pre_emotions = np.zeros(emotion_type)

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

for pre_file in pre_file_list:
    pre_file_path = os.path.join(pre_labels_path, pre_file)
    list_f = pd.read_csv(pre_file_path)
    pre_labels = list_f['emotion']
    for i in range(emotion_type):
        pre_emotions[i] = sum(pre_labels == i)
    emotion_mark = pre_file.split('.')[0]
    plt.title(emotion_mark)
    plt.plot([0,1,2,3,4,5,6], pre_emotions, 'o')
    plt.show()

#    print('Correct emotion: ' + pre_file.split('.')[0] + '\n')
#    print('Pre_emotion: ' + reindex_labels(np.argmax(np.array(pre_labels))) + '\n')
