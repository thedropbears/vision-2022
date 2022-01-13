from camera_manager import CameraManager
from connection import NTConnection
import cv2
import numpy as np
import time
from typing import Tuple, Optional

class Vision:
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection

    def run(self):
        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """

        frame_time, self.frame = self.camera_manager.get_frame()
        # frame time is 0 in case of an error
        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return
        
        # Flip the image beacuse it's originally upside down.
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        results = self.process_image(self.frame)

        if results is not None:
            distance, angle = results
            self.connection.send_results(
                (distance, angle, time.monotonic())
            )  # distance (meters), angle (radians), timestamp

        # send image to display on driverstation
        self.camera_manager.send_frame(self.frame)
        self.connection.set_fps()

    def process_image(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Takes a frame returns the target dist & angle
        returns None if there is no target
        """

        new_frame = self.preprocess(frame)
        contours = self.find_contours(new_frame)
        filtered_contours = self.filter_contours(contours)
        if len(filtered_contours) == 0:
            return None
        groups = self.group_contours(filtered_contours)
        if len(groups) == 0:
            return None
        best_group = self.rank_groups(groups)
        dist, angle = self.get_values(best_group)

        return dist, angle

    def preprocess(self, frame):
        """Do any preprocessing we might want to do on the frame
        e.g. convert to hsv, threshold, erode/dilate
        """
        return frame

    def find_contours(self, frame):
        """Finds the contours, see https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        """
        return []

    def filter_contours(self, contours):
        """Takes contours and returns on ones that are likely to be vision tape
        Checks things like size, width to height ratio
        """
        return []

    def group_contours(self, contours):
        """Create potential groups out of the contours
        Group based on distance, sizes and relative positions
        """
        pass

    def rank_groups(self, groups):
        """Returns the group that is most likely to be the target
        Could use metrics such as curvature, consistency of contours, consistency with last frame
        """
        pass

    def get_values(self, group):
        """Get angle and distance from the camera of the group
        Find the center of the group and then do maths to work out angle and dist
        """
        pass


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py

    vision = Vision(
        CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
        NTConnection(),
    )
    while True:
        vision.run()