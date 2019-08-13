import os
import numpy as np

def str2emotion_index(s):
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    for i, e in enumerate(emotions):
        if e in s:
            return i

    raise ValueError('Input string does not contain any emotions!')


def read_result(root_dir):
    ret = {}
    for person_dir in sorted(os.listdir(root_dir)):
        if not '_short_movie' in person_dir:
            continue

        result_of_person = np.zeros((7, 7))

        for emotion_csv in sorted(os.listdir(os.path.join(root_dir, person_dir, 'result'))):
            if not '.cvs' in emotion_csv:
                continue

            label_index = str2emotion_index(emotion_csv)

            with open(os.path.join(root_dir, person_dir, 'result', emotion_csv), 'r') as f:
                content = f.readlines()

            for line in content[1:]:
                predict = int(line.split(',')[0])
                result_of_person[label_index][predict] += 1

        ret[person_dir] = result_of_person

    return ret
