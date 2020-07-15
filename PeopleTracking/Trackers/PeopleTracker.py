import cv2
import dlib
import numpy as np


class PeopleTracker:
    '''
    Class for tracking detected people using correlation tracker.
    '''
    def __init__(self):
        self.trackers = []
        self.rectangles = []
        self.persons_images = []

    def start_tracking(self, frame, rectangles):
        self.trackers = []
        # Convert the frame from BGR to RGB for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for rectangle in rectangles:
            # Construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(rectangle[0], rectangle[1], rectangle[2], rectangle[3])
            tracker.start_track(rgb, rect)
            self.trackers.append(tracker)

    def update_tracking(self, frame):
        # Convert the frame from BGR to RGB for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rectangles = []
        self.persons_images = []

        for tracker in self.trackers:
            tracker.update(rgb)
            position = tracker.get_position()
            start_x = int(position.left())
            start_y = int(position.top())
            end_x = int(position.right())
            end_y = int(position.bottom())
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            self.rectangles.append(np.array([start_x, start_y, end_x, end_y], dtype="int32"))
            self.persons_images.append(frame[start_y:end_y, start_x:end_x])

        return self.rectangles, self.persons_images
