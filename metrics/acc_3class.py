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


def main(root_dir):
    result_per_person = read_result(root_dir)

    result_all_person = np.sum(list(result_per_person.values()), axis=0)
    print('result_all_person')
    print(result_all_person)

    class_num = len(class_split)
    ret = np.zeros((class_num, class_num))

    for label_3, label_indexs in enumerate(class_split.values()):
        for predict_3, predict_indexs in enumerate(class_split.values()):
            for label_7 in label_indexs:
                for predict_7 in predict_indexs:
                    ret[label_3][predict_3] += result_all_person[label_7][predict_7]

    print('result 3 class')
    print(ret)
    
    ret_list = {'true': [], 'pred': []}
    for label in range(class_num):
        for predict in range(class_num):
            ret_list['true'].extend([label]*int(ret[label][predict]))
            ret_list['pred'].extend([predict]*int(ret[label][predict]))

    print(classification_report(ret_list['true'], ret_list['pred'], labels=class_split.keys()))



if __name__ == '__main__':
    main(sys.argv[1])