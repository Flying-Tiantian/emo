import os
import numpy as np


def str2emotion_index(s):
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


def read_result(root_dir):
    ret = {}
    for person_dir in sorted(os.listdir(root_dir)):
        if not '_short_movie' in person_dir:
            continue

        result_of_person = np.zeros((3, 3), dtype=int)

        result_dir_name = 'result'
        for files in sorted(os.listdir(os.path.join(root_dir, person_dir))):
            if 'result' in files:
                result_dir_name = files
                break

        for emotion_csv in sorted(os.listdir(os.path.join(root_dir, person_dir, result_dir_name))):
            if not emotion_csv.endswith('.cvs'):
                continue

            label_index = str2emotion_index(emotion_csv)

            print('Reading %s ...' % os.path.join(root_dir, person_dir, result_dir_name, emotion_csv))
            with open(os.path.join(root_dir, person_dir, result_dir_name, emotion_csv), 'r') as f:
                content = f.readlines()

            for line in content[1:]:
                predict = int(line.split(',')[0])
                result_of_person[label_index][predict] += 1

        ret[person_dir] = result_of_person

    return ret
