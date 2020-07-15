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
import cv2


class AgeAndGenderEstimator():
    """
    Class for face detection, cropping face image from the frame, estimating age and gender based on pre-trained models.
    """
    def __init__(self):
        """
        Initialisation of object for age and gender estimation for one person.
        """
        # Load pre-trained models and put their corresponding classes in lists.
        self.age_model = cv2.dnn.readNetFromCaffe("./MobilenetSSD/age_deploy.prototxt", "./MobilenetSSD/age_net.caffemodel")
        self.age_list = [1, 2, 3, 4, 5, 6, 7, 8]

        self.gender_model = cv2.dnn.readNetFromCaffe("./MobilenetSSD/gender_deploy.prototxt", "./MobilenetSSD/gender_net.caffemodel")
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

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        self.gender_model.setInput(blob)
        gender_predictions = self.gender_model.forward()
        gender = self.gender_list[gender_predictions.argmax()]

        self.age_model.setInput(blob)
        age_predictions = self.age_model.forward()
        age = self.age_list[age_predictions.argmax()]

        # label = "{},{}".format(gender, age)
        return age, gender
