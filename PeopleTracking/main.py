#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Nemanja Krajcinovic
# email: kr.nemanjja@gmail.com
#
#
# People detecting using CNN pre-trained model and tracking while they are
# inside the frame boundings.
#
# usage: The script is pre-configured to load pre-train model.
#
# changes:
# - 15.5.2020. - Nemanja Krajcinovic
# - Pre-trained models downloaded. Improvements in detection.
# - 21.5.2020. - Nemanja Krajcinovic
# - Created object tracking using centroid tracker of every person.
#
############################################################################
from Trackers.CentroidTracker import CentroidTracker
from Trackers.BirdEyeView import BirdEyeView
# MobilNetSDD
# from Trackers.PeopleDetector import PeopleDetector
# YOLO-COCO
from Trackers.PeopleDetectorYOLO import PeopleDetector
from Trackers.PeopleTracker import PeopleTracker
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, putText, FONT_HERSHEY_PLAIN, imshow, waitKey, getWindowProperty, WND_PROP_VISIBLE, destroyAllWindows
from imutils.video import FileVideoStream
from imutils.video import FPS


def main():
    '''
    Detecting and tracking people, and showing their trajectory in bird-eye view.
    Drawing bounding boxes and corresponding IDs around every person on every frame.
    '''
    # Initialise people detector.
    # MobileNetSSD
    # people_detector = PeopleDetector()

    # YOLO-COCO
    people_detector = PeopleDetector(use_gpu=True)

    # Initialise people tracker.
    people_tracker = PeopleTracker()

    # Initialise people tracker.
    centroid_tracker = CentroidTracker(set_max_disappear=20, set_max_distance=100)

    # Initialise bird-view transformation object.
    bird_eye_transform = BirdEyeView()

    persons = list()

    # Initialise the video stream and frame dimensions.
    # 1. Basic load of video stream.
    # capture = VideoCapture(0)
    # capture = VideoCapture("./conf/TestVideo4.avi")
    # capture = VideoCapture("rtsp://admin:saga12345@192.168.0.108:554/")
    # (W, H) = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 2. Extended video stream with que, which size is 128 frames, and it is used to enable longer period for processing every frame.
    # fvs = FileVideoStream(0).start()
    # fvs = FileVideoStream("./conf/TestVideo25FPS.mp4").start()
    fvs = FileVideoStream("rtsp://admin:saga12345@172.24.0.21:554/").start()
    (W, H) = (1280, 720)

    # Initialise object for determining how much frames are processed per one second.
    fps = FPS().start()

    # Counter of frames which wont be used for detection. Instead of it, these frames will be used for tracking.
    total_frames = 0

    # 1. Basic video stream receiving.
    '''
    while capture.isOpened():
        _, frame = capture.read()

    '''

    # 2. Qued video stream receiving.
    while fvs.more():
        frame = fvs.read()

        # Initialise the list of bounding box rectangles returned by either (1) our object detector or (2) the correlation trackers
        rectangles = []

        if total_frames % 10 == 0:
            # Lists of bounding box rectangles of detected people along with cropped pictures of detected people.
            rectangles, persons_images = people_detector.update(frame, W, H)

            people_tracker.start_tracking(frame, rectangles)
        else:
            rectangles, persons_images = people_tracker.update_tracking(frame)

        # Update centroid tracker.
        persons = centroid_tracker.update(rectangles, persons_images, persons)

        for i in range(0, len(persons)):
            persons[i].draw_on_frame(frame)

        for i in range(0, len(persons)):
            persons[i].persons_transformed_downoid = bird_eye_transform.compute_transformed_point(persons[i].persons_downoid)
            bird_eye_transform.draw_on_floor(persons[i].persons_trajectory, persons[i].color)

        bird_eye_transform.draw_floor_bounding_lines(frame)

        # Show the output frame.

        # 2. Addition to show how full is que when using extended video stream receiver.
        putText(frame, "Queue occupancy [max 128]: {}".format(fvs.Q.qsize()), (10, 30), FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

        imshow("Video", frame)
        waitKey(1)
        if getWindowProperty("Video", WND_PROP_VISIBLE) < 1:
            break
        total_frames += 1
        fps.update()

    # Cleanup of opened windows and video-stream.
    fps.stop()
    print("Approximate FPS: {:.2f}".format(fps.fps()))
    destroyAllWindows()
    # capture.release()
    fvs.stop()


if __name__ == "__main__":
    main()
