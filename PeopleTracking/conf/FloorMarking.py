import cv2
import yaml


class InitialConfiguration:
    '''
    Class for initial configuration: marking floor with 4 points.
    '''
    def __init__(self):
        self.list_points = list()

        self.capture = cv2.VideoCapture("rtsp://admin:saga12345@192.168.0.108:554/")
        # self.capture = cv2.VideoCapture("./TestVideo25FPS.mp4")

        self.windowName = 'Floor Marking'
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName, self.CallBackFunc)

    def CallBackFunc(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Left button of the mouse is clicked - position (", x, ", ", y, ")")
            self.list_points.append([x, y])

    def configure(self):
        print("Floor marking.")
        print("Please mark 4 points which surround the floor, in the next order: ")
        print("1. Top left point")
        print("2. Top right point")
        print("3. Bottom right point")
        print("4. Bottom left point")

        while self.capture.isOpened():
            _, frame = self.capture.read()
            cv2.imshow(self.windowName, frame)

            if len(self.list_points) == 4:
                config_data = dict(floor_points=dict(top_left=self.list_points[0], top_right=self.list_points[1], bottom_right=self.list_points[2], bottom_left=self.list_points[3]))

                with open('./floor_points.yml', 'w') as outfile:
                    yaml.dump(config_data, outfile, default_flow_style=False)
                    cv2.imwrite("./floor.jpg", frame)
                    break

            cv2.waitKey(1)
            if cv2.getWindowProperty("Floor Marking", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.capture.release()


if __name__ == "__main__":
    floor_marking = InitialConfiguration()
    floor_marking.configure()
