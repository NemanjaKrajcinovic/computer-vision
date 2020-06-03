#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
# Pose estimation of chess board.
#
# usage: As example, 3D coord system is put onto chess table and drawn.
#
# changes:
# - 10.4.2020. – Natasa Avramovic
# - Function created
#
############################################################################
import numpy as np
from cv2 import *
from CameraCalibrationChessboard.ChessboradCameraCalibration import camera_calibration_chessboard


def draw(img, corners, imgpts):
    '''
    Function for drawing lines
    '''
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def pose_estimation_chess():

    # Creating prepared 3D camera coordinate system points
    prepared_object_points = np.zeros((7 * 7, 3), np.float32)
    prepared_object_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    camera_matrix, dist_coeffs, _, _, mapx, mapy = camera_calibration_chessboard()

    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        _, frame = cap.read()
        if mapx is not None and mapy is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                ret, rvec, tvec = cv2.solvePnP(prepared_object_points, corners2, camera_matrix, dist_coeffs)
                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                frame = draw(frame, corners2, imgpts)

        cv2.imshow('CalibratedCamera', frame)
        cv2.waitKey(1)
        if cv2.getWindowProperty('CalibratedCamera', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
