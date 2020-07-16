#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
# Age and Gender estimation using CNN pre-trained models from Intel OpenVINO.
#
# usage: Face detection using OpenCV dnn module with pre-trained model.
#        Age and Gender estimation using OpenCV dnn module with pre-trained model.
#
# changes:
# - 5.5.2020. - Natasa Avramovic
# - Pre-trained model for face detection updated
#
############################################################################
import cv2


class AgeAndGenderEstimator():
    """
    Class for face detection, cropping face image from the frame, estimating age and gender based on pre-trained models.
    """
    def __init__(self):
        """
        Initialisation of object for age and gender estimation for one person.
        """
        # Load pre-trained model and put their corresponding classes in lists.
        self.model = cv2.dnn.Net_readFromModelOptimizer("./MobilenetSSD/age-gender-recognition-retail-0013.xml", "./MobilenetSSD/age-gender-recognition-retail-0013.bin")
        self.gender_list = [1, 2]

        self.face_model = cv2.dnn.readNetFromCaffe("./MobilenetSSD/deploy.prototxt", "./MobilenetSSD/res10_300x300_ssd_iter_140000.caffemodel")

        self.padding = 5

    def get_face_box(self, frame, confidence_threshold=0.7):
        '''
        Face detection and calculation of bounding box rectangle around face.
        Returns bounding box of the detected face.
        '''
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_model.setInput(blob)
        detections = self.face_model.forward()
        bounding_box = []

        if detections[0, 0, 0, 2] > confidence_threshold:
            start_x = int(detections[0, 0, 0, 3] * frame_width)
            start_y = int(detections[0, 0, 0, 4] * frame_height)
            end_x = int(detections[0, 0, 0, 5] * frame_width)
            end_y = int(detections[0, 0, 0, 6] * frame_height)
            bounding_box = [start_x, start_y, end_x, end_y]
        return bounding_box

    def estimate_age_and_gender(self, frame):
        '''
        Estimation of age and gender on detected face.
        Returns string in format gender, age.
        '''
        bounding_box = self.get_face_box(frame)

        if not bounding_box:
            return 0, 0

        face = frame[max(0, bounding_box[1] - self.padding):min(bounding_box[3] + self.padding, frame.shape[0] - 1), max(0, bounding_box[0] - self.padding):min(bounding_box[2] + self.padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
        self.model.setInput(blob)
        detections = self.model.forwardAndRetrieve(['prob', 'age_conv3'])

        gender = self.gender_list[detections[0][0][0].argmax()]
        age = detections[1][0][0][0][0][0] * 100

        # label = "{},{}".format(gender, age)
        return age, gender
