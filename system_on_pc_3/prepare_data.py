import os
import cv2
import numpy as np



def read_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            return
        yield frame, video_capture.get(cv2.cv.CV_CAP_PROP_POS_MSEC)


def crop_eye(image, crop_params):
    left, up, size = crop_params
    height, width, channel = np.shape(image)
    left = int(width * left)
    up = int(height * up)
    size = int(width * size)

    # gray_image = image[:,:,0]  # 红外摄像头，三通道的值相同
    
    return image[up: up+size, left: left+size, :]


def get_eye_images(video_path, crop_params, get_time=False):
    for frame, time in read_video(video_path):
        if get_time:
            yield crop_eye(frame, crop_params), time
        else:
            yield crop_eye(frame, crop_params)
