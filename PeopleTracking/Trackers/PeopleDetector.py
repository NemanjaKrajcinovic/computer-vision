import numpy as np
from cv2 import dnn, resize, rectangle


class PeopleDetector:
    '''
    Class for detection persons and returning bounding boxes of every detected person.
    '''
    model = dnn.readNetFromCaffe("./MobilenetSSD/MobileNetSSD_deploy.prototxt", "./MobilenetSSD/MobileNetSSD_deploy.caffemodel")
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "bike", "person", "plant", "sheep", "sofa", "train", "tv"]

    def __init__(self):
        self.rectangles = []
        self.persons_images = []

    def update(self, frame, W, H):
        # Construct a blob from the frame, pass it through the network, obtain output predictions.
        blob = dnn.blobFromImage(resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.model.setInput(blob)
        detections = self.model.forward()

        # Initialise the list of bounding box rectangles.
        self.rectangles = []
        self.persons_images = []

        # Loop over the detections.
        for i in range(0, detections.shape[2]):
            # Filter out predictions which are not persons.
            if self.classes[int(detections[0, 0, i, 1])] != "person":
                continue

            # Filter out weak detections.
            if detections[0, 0, i, 2] > 0.5:
                # Compute the (x, y) coordinates of the bounding box for the person.
                bounding_box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (start_x, start_y, end_x, end_y) = bounding_box.astype("int")

                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = W if end_x > W else end_x
                end_y = H if end_y > H else end_y

                # Update the bounding box rectangles list
                self.rectangles.append(bounding_box.astype("int"))
                self.persons_images.append(frame[start_y:end_y, start_x:end_x])

                rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return self.rectangles, self.persons_images
