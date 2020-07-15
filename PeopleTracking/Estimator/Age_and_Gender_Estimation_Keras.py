#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
# Age and Gender estimation using CNN pre-trained models.
#
# usage: Face detection using OpenCV dnn module with pre-trained model.
#        Age and Gender estimation using OpenCV dnn module with pre-trained model.
#
# changes:
# - 5.5.2020. - Natasa Avramovic
# - Pre-trained model for face detection updated
#
############################################################################
from cv2 import dnn, resize, INTER_AREA
from os import path, getcwd
import numpy as np
from Estimator.WideResnet import WideResNet
from keras.utils.data_utils import get_file


class AgeAndGenderEstimator():
    """
    Class for face detection, cropping face image from the frame, estimating age and gender based on pre-trained models.
    """
    model = WideResNet(64, depth=16, k=8)()
    face_size = 64
    face_model = dnn.readNetFromCaffe("./MobilenetSSD/deploy.prototxt", "./MobilenetSSD/res10_300x300_ssd_iter_140000.caffemodel")
    padding = 30

    def __init__(self):
        fpath = get_file('weights.18-4.06.hdf5', ".\pretrained_models\weights.18-4.06.hdf5", cache_subdir=path.join(getcwd(), "pretrained_models"))
        self.model.load_weights(fpath)

    def get_face_box(self, frame):
        '''
        Face detection and calculation of bounding box rectangle around face.
        Returns bounding box of the detected face.
        '''
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_model.setInput(blob)
        detections = self.face_model.forward()
        bounding_box_of_face = []

        if detections[0, 0, 0, 2] > 0.99:
            start_x = int(detections[0, 0, 0, 3] * frame_width)
            start_y = int(detections[0, 0, 0, 4] * frame_height)
            end_x = int(detections[0, 0, 0, 5] * frame_width)
            end_y = int(detections[0, 0, 0, 6] * frame_height)
            bounding_box_of_face = [start_x, start_y, end_x, end_y]
            return bounding_box_of_face
        else:
            return None

    def estimate_age_and_gender(self, frame):
        '''
        Estimation of age and gender on detected face.
        Returns string in format gender, age.
        '''
        bounding_box_of_face = self.get_face_box(frame)

        if bounding_box_of_face is None:
            return None, None

        face = frame[max(0, bounding_box_of_face[1] - self.padding):min(bounding_box_of_face[3] + self.padding, frame.shape[0] - 1), max(0, bounding_box_of_face[0] - self.padding):min(bounding_box_of_face[2] + self.padding, frame.shape[1] - 1)]
        resized_face = resize(face, (64, 64), interpolation=INTER_AREA)
        resized_face = np.array(resized_face)

        face_imgs = np.empty((1, 64, 64, 3))
        face_imgs[0, :, :, :] = resized_face

        predictions = self.model.predict(face_imgs)

        gender_list = predictions[0]
        gender = gender_list[0][0]

        age_list = np.arange(0, 101).reshape(101, 1)
        age = predictions[1].dot(age_list).flatten()

        return age, gender
