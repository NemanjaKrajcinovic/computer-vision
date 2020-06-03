#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Nemanja Krajcinovic
# email: kr.nemanjja@gmail.com
#
# Camera calibration using chess board.
#
# usage: Based on finding corners on the frame of the video stream,
#        the function calculates intrinsic paramters of a camera and its
#        distorsion coeffs.
#
# requirements to the Test Cases:
#    Chess table must be close enough to the camera that camera can recognize
#    every ArUco marker.
# changes:
# - 19.4.2020. – Natasa Avramovic
# - Added refining of coreners
# - 4.4.2020. – Nemanja Krajcinovic
# - Initially created
#
############################################################################
import numpy as np
from cv2 import *


def camera_calibration_chessboard():

    cap = cv2.VideoCapture(0)

    # Creating object_points which represent 3D camera coordinate system,
    # and image_points where points from 2D image will be stored
    object_points = []
    image_points = []

    # Creating prepared 3D camera coordinate system points
    prepared_object_points = np.zeros((7 * 7, 3), np.float32)
    prepared_object_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    number_of_pictures_for_calibration = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while(cap.isOpened()):
        _, frame = cap.read()
        cv2.imshow('Video', frame)
        k = cv2.waitKey(1)

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord(' '):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

            if ret:
                object_points.append(prepared_object_points)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners)

                cv2.drawChessboardCorners(frame, (7, 7), refined_corners, ret)
                cv2.imshow('PictureWithCorners', frame)
                number_of_pictures_for_calibration += 1

        elif number_of_pictures_for_calibration == 15:
            break

    cap.release()
    cv2.destroyAllWindows()

    err, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    print('Error: ', err)

    # Generating the corrections
    new_camera_matrix, valid_pix_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray.shape[::-1], 0)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, gray.shape[::-1], cv2.CV_32FC1)

    return camera_matrix, dist_coeffs, new_camera_matrix, valid_pix_roi, mapx, mapy
