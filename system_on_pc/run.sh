#!/bin/sh

python convert_video_to_pic.py
python crop_eye.py
rm pic_crop_zy/list.txt
python extract_feature_7_pics.py
python from_video_classify_emotion.py
