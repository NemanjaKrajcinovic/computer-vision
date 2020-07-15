'''
Created on 14 Jul 2020

@author: Nemanja
'''
import numpy as np
from cv2 import dnn, rectangle
import os


class PeopleDetector:
    '''
     Class for detection persons and returning bounding boxes of every detected person.
     YOLO detector, advance for using NVIDIA CUDA engine.
    '''
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO
    layer_names = net.getLayerNames()

    def __init__(self, use_gpu):
        '''
        Constructor
        '''
        PeopleDetector.layer_names = [PeopleDetector.layer_names[i[0] - 1] for i in PeopleDetector.net.getUnconnectedOutLayers()]
        # check if we are going to use GPU
        if use_gpu:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            PeopleDetector.net.setPreferableBackend(dnn.DNN_BACKEND_CUDA)
            PeopleDetector.net.setPreferableTarget(dnn.DNN_TARGET_CUDA)

    def update(self, frame, W, H):
        # Construct a blob from the frame, pass it through the network, obtain output predictions.
        blob = dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        PeopleDetector.net.setInput(blob)
        layer_outputs = PeopleDetector.net.forward(PeopleDetector.layer_names)

        # Initialise the list of bounding box rectangles.
        self.boxes = []
        self.confidences = []
        self.rectangles = []
        self.persons_images = []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter detections by (1) ensuring that the object
                # detected was a person and (2) that the minimum
                # confidence is met
                if classID == 0 and confidence > 0.3:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # update our list of bounding box coordinates,
                    # centroids, and confidences
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = dnn.NMSBoxes(self.boxes, self.confidences, 0.3, 0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                (start_x, start_y, end_x, end_y) = (x, y, x + w, y + h)

                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = W if end_x > W else end_x
                end_y = H if end_y > H else end_y

                bounding_box = (int(start_x), int(start_y), int(end_x), int(end_y))

                # Update the bounding box rectangles list
                self.rectangles.append(bounding_box)
                self.persons_images.append(frame[start_y:end_y, start_x:end_x])

                rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return self.rectangles, self.persons_images
