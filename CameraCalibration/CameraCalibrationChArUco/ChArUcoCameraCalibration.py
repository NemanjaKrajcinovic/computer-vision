#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
# Camera calibration using ChArUco board.
#
# usage: Based on finding every ArUco marker on the frame of the video stream,
#        the function calculates intrinsic paramters of a camera and its
#        distorsion coeffs.
#
#
# requirements to the Test Cases:
#    ChArUco table must be close enough to the camera that camera can recognize
#    every ArUco marker.
# changes:
# - 20.4.2020. – Natasa Avramovic
# - Set every fifth frame to be processed
# - 14.4.2020. – Nemanja Krajcinovic
# - Added calculation of undistorted image in whole function
#
############################################################################
import cv2.cv2 as cv2


def camera_calibration_charuco():
    '''
    Camera calibration using ChArUco board predefined in dictionary.
    '''
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5, 7, .025, .0125, dictionary)
    detectorParams = cv2.aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(0)

    # Number of ArUco markers which must be detected in whole process
    required_count = 50

    all_corners = []
    all_ids = []

    # Iterators for selecting every fifth frame to be processed
    frame_idx = 0
    frame_spacing = 5

    while(cap.isOpened()):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, detectorParams)

        if len(marker_corners) > 0 and frame_idx % frame_spacing == 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

            cv2.aruco.drawDetectedMarkers(gray, marker_corners, marker_ids)

        cv2.imshow('Calibration', gray)
        cv2.waitKey(1)
        if cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_idx += 1
        print("Found: " + str(len(all_ids)) + " / " + str(required_count))

        if len(all_ids) >= required_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Finished collecting data, computing...')
    err, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape[::-1], None, None)
    print('Error: ', err)

    # Generating the corrections
    new_camera_matrix, valid_pix_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray.shape[::-1], 0)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, gray.shape[::-1], cv2.CV_32FC1)

    return camera_matrix, dist_coeffs, new_camera_matrix, valid_pix_roi, mapx, mapy
