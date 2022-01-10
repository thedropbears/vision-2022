from camera_manager import CameraManager


class Vision:
    def __init__(self, camera_manager):
        pass

    def run(self, frame):
        contours = self.find_contours(frame)
        filtered_contours = self.filter_contours(contours)
        groups = self.group_contours(filtered_contours)
        best_group = self.find_group(groups)
        azimuth, tilt = self.get_values(best_group)

        return azimuth, tilt

    def find_contours(self, frame):
        pass

    def filter_contours(self, contours):
        pass

    def group_contours(self, contours):
        pass

    def find_group(self, groups):
        pass

    def get_values(self, group):
        pass


if __name__ == "__main__":
    # code to run on the robot
    pass
