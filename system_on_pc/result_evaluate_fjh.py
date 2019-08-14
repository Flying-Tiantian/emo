import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

pre_labels_path = './result_zy/'
fig_save_path = os.path.join(pre_labels_path, 'fig')
emotion_type = 7
pre_file_list =sorted(os.listdir(pre_labels_path))
pre_emotions = np.zeros(emotion_type)

if os.path.exists(fig_save_path):
    shutil.rmtree(fig_save_path)
os.mkdir(fig_save_path)

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
    if not len(pre_file.split('.')) == 2:
        continue
    pre_file_path = os.path.join(pre_labels_path, pre_file)
    list_f = pd.read_csv(pre_file_path)
    pre_labels = list_f['emotion']
    weights = list_f['weights']
    for i in range(emotion_type):
        pre_emotions[i] = sum(weights[pre_labels == i] * weights[pre_labels == i])
    emotion_mark = pre_file.split('.')[0]
    plt.title(emotion_mark)
    plt.plot([0,1,2,3,4,5,6], pre_emotions, 'o')
    plt.savefig(os.path.join(fig_save_path, emotion_mark + '.png'))
    plt.cla()
#    plt.show()

#    print('Correct emotion: ' + pre_file.split('.')[0] + '\n')
#    print('Pre_emotion: ' + reindex_labels(np.argmax(np.array(pre_labels))) + '\n')
