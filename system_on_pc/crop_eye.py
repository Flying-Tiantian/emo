import numpy as np
import cv2
import os
import shutil
import skimage.color

DATA_PATH = './pic_zy/'
SAVE_PATH = './pic_crop_zy/'
TXT_NAME = 'list.txt'

eye_size = 224

def crop_eye_out(I):

  # modify here for croping eye -------------------------------------
  # for happy.mov
  gap = 850
  I = I[60:60+gap, 500:500+gap]
#  gap = I.shape[1]*5/8
#  I = I[I.shape[0]*1/16:I.shape[0]*1/16 + gap, I.shape[1]*2/16:I.shape[1]*2/16 + gap]
  gray_I = I[:,:,0]
  I[:,:,1] = gray_I
  I[:,:,2] = gray_I
  return I

def process_single_pic(pic_path):
  im = cv2.imread(pic_path)
  im = crop_eye_out(im)
  return im 

def crop_eye():
  f = open(os.path.join(SAVE_PATH, TXT_NAME), 'wb')
  pic_list = open(os.path.join(DATA_PATH, TXT_NAME))
  for i, line in enumerate(pic_list):
    label = line.split(' ')[1]
    pic = line.split(' ')[0]
    pic_path = os.path.join(DATA_PATH, pic)
    im = process_single_pic(pic_path)
#    im = skimage.color.rgb2gray(im)
#    im = cv2.resize(im, (eye_size, eye_size))
    target_file = os.path.join(SAVE_PATH, pic)
    cv2.imwrite(target_file, im)
    f.write(line)


if os.path.exists(SAVE_PATH):
  shutil.rmtree(SAVE_PATH)

os.mkdir(SAVE_PATH)
crop_eye()

