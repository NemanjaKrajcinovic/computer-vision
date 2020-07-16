import numpy as np
from cv2 import imread, circle, imshow, getPerspectiveTransform, warpPerspective, perspectiveTransform, polylines, LINE_AA
import yaml


class BirdEyeView:
    '''
    Class for bird-eye view transformations.
    '''
    corner_points = []
    matrix = []
    floor = []

    def __init__(self):
        """
        Initialisation of object for bird-eye view transformations.
        """
        # Load the configuration for the top-down view
        with open("./conf/floor_points.yml", "r") as ymlfile:
            points = yaml.load(ymlfile, Loader=yaml.BaseLoader)

        BirdEyeView.corner_points.append(points["floor_points"]["top_left"])
        BirdEyeView.corner_points.append(points["floor_points"]["top_right"])
        BirdEyeView.corner_points.append(points["floor_points"]["bottom_right"])
        BirdEyeView.corner_points.append(points["floor_points"]["bottom_left"])

        # Compute the transformation matrix and return it, along with transformed picture of the floor.
        self.compute_perspective_transform(imread("./conf/floor.jpg"))
        # blank_image = np.zeros((self.height, self.width, 3), np.uint8)

    def draw_on_floor(self, persons_trajectory, color):
        for i in range(0, len(persons_trajectory)):
            circle(self.floor, persons_trajectory[i], 3, color, -1)
        polylines(self.floor, [np.array(persons_trajectory, dtype="int32").reshape(-1, 1, 2)], False, color, 2, lineType=LINE_AA)
        imshow("Bird-eye view", self.floor)

    def compute_perspective_transform(self, image):
        """ Compute the transformation matrix.
        @ corner_points : 4 corner points selected from the image.
        @ image: original image which will be cropped and warped in order to get picture of the floor.
        """
        rect = np.array(BirdEyeView.corner_points, dtype="float32")
        (top_left, top_right, bottom_right, bottom_left) = rect

        # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left or the top-right and top-left points.
        width = max(int(np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))), int(np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))))

        # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right or the top-left and bottom-left points.
        height = max(int(np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))), int(np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))))

        # Construct the set of destination points to obtain a "Bird-eye view".
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # Compute the perspective transform matrix and then apply it.
        BirdEyeView.matrix = getPerspectiveTransform(rect, dst)
        BirdEyeView.floor = warpPerspective(image, BirdEyeView.matrix, (width, height))

    def compute_transformed_point(self, downoid):
        """ Apply the perspective transformation to every ground point which have been detected on the main frame.
        @ downoids_list : List that contains the points to transform.
        return : List containing all the new points.
        """
        # Compute the new coordinates of points.
        downoid_reshaped = np.array(downoid, dtype="float32").reshape(-1, 1, 2)
        transformed_downoid = perspectiveTransform(downoid_reshaped, BirdEyeView.matrix)
        transformed_downoid_reshaped = (int(transformed_downoid[0][0][0]), int(transformed_downoid[0][0][1]))

        return transformed_downoid_reshaped

    def draw_floor_bounding_lines(self, image):
        '''
        Draw rectangle box over the delimitation area.
        '''
        polylines(image, [np.array(BirdEyeView.corner_points, dtype="int32").reshape(-1, 1, 2)], True, (255, 0, 0), 2, lineType=LINE_AA)
