#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
# Create ChArUco board.
#
# usage: In order to calibrate camera as better as possible, ChArUco table
#        must be created using this function on printed on white paper.
#
# requirements to the Test Cases:
#    ChArUco table must be printed and sealed up for some flat surface
#    because table must remain flat.
# changes:
# - 4.4.2020. – Nemanja Krajcinovic
# - Set table size 5x7
#
############################################################################
from cv2 import *

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(5, 7, .025, .0125, dictionary)
img = board.draw((200 * 5, 200 * 7))
detectorParams = cv2.aruco.DetectorParameters_create()
cv2.imwrite('charuco.png', img)
