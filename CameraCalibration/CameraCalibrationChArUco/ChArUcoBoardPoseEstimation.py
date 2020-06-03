#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Nemanja Krajcinovic
# email: kr.nemanjja@gmail.com
#
# Pose estimation of ChArUco board.
#
# usage: As example, 3D coordinate system is put onto ChArUco table and drawn.
#
# changes:
# - 10.4.2020. – Nemanja Krajcinovic
# - Function created
#
############################################################################
from cv2 import *
from CameraCalibrationChArUco.ChArUcoCameraCalibration import camera_calibration_charuco


def pose_estimation_charuco():

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5, 7, .025, .0125, dictionary)
    detectorParams = cv2.aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        _, frame = cap.read()
        camera_matrix, dist_coeffs, _, _, mapx, mapy = camera_calibration_charuco()
        if mapx is not None and mapy is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, dictionary, detectorParams)

            if len(marker_corners) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, board)

                if charuco_ids is not None:
                    valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

                    if valid:
                        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        cv2.imshow('CalibratedCamera', frame)
        cv2.waitKey(1)
        if cv2.getWindowProperty('CalibratedCamera', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
