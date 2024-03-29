import os
import sys
import numpy as np
import pprint
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from read_csv import read_result


emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
class_split = {'positive': [0, 1, 2, 4], 
               'negative': [3], 
               'neutral': [6]}


def main_7_to_3(root_dir):
    result_per_person = read_result(root_dir)

    result_all_person = np.sum(list(result_per_person.values()), axis=0)

    class_num = len(class_split)
    ret = np.zeros((class_num, class_num), dtype=int)

    for label_3, label_indexs in enumerate(class_split.values()):
        for predict_3, predict_indexs in enumerate(class_split.values()):
            for label_7 in label_indexs:
                for predict_7 in predict_indexs:
                    ret[label_3][predict_3] += result_all_person[label_7][predict_7]
    
    ret[0][2] = 0
    ret[1][2] = 0

    ret_list = {'true': [], 'pred': []}
    for label in range(class_num):
        for predict in range(class_num):
            ret_list['true'].extend([label]*int(ret[label][predict]))
            ret_list['pred'].extend([predict]*int(ret[label][predict]))

    
    print(ret)

    print(classification_report(ret_list['true'], ret_list['pred'], target_names=class_split.keys()))


def main_3(root_dir):
    result_per_person = read_result(root_dir)

    result_all_person = np.sum(list(result_per_person.values()), axis=0)
    class_num = len(result_all_person)

    ret = result_all_person
    ret[0][2] = 0
    ret[1][2] = 0

    ret_list = {'true': [], 'pred': []}
    for label in range(class_num):
        for predict in range(class_num):
            ret_list['true'].extend([label]*int(ret[label][predict]))
            ret_list['pred'].extend([predict]*int(ret[label][predict]))

    
    print(ret)

    print(classification_report(ret_list['true'], ret_list['pred'], target_names=class_split.keys()))


if __name__ == '__main__':
    main_3(sys.argv[1])