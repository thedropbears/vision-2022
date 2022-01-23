from camera_manager import CameraManager
from connection import NTConnection
from magic_numbers import *
from typing import Tuple, Optional, List
import cv2
import numpy as np
import time

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
            x, y = results
            self.connection.send_results(
                (x, y, time.monotonic())
            )  # x and y in NDC, positive axes right and down; timestamp

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
        self.contourAreas = [cv2.contourArea(c) for c in filtered_contours]
        groups = self.group_contours(filtered_contours)
        if len(groups) == 0:
            return None
        best_group = self.rank_groups(filtered_contours, groups)
        if best_group is None:
            return None
        return self.get_values(filtered_contours, best_group, (frame.shape[1], frame.shape[0]))

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Creates a mask of expected target green color
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(hsv, TARGET_HSV_LOW, TARGET_HSV_HIGH)
        self.mask = cv2.erode(self.mask, ERODE_DILATE_KERNEL)
        self.mask = cv2.dilate(self.mask, ERODE_DILATE_KERNEL)
        return self.mask

    def find_contours(self, mask: np.ndarray) -> np.ndarray:
        """Finds contours on a grayscale mask
        """
        *_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            self.frame,
            contours,
            -1,
            (255, 0, 0),
            thickness=2,
        )
        return contours

    def filter_contours(self, contours: np.ndarray) -> np.ndarray:
        """Unimplemented, returns identity
        """
        return contours

    def group_contours(self, contours: np.ndarray) -> List[List[int]]:
        """Returs a nested list of contour indices grouped by relative distance
        """
        # Build an adjacency list based on distance between contours relative to their area
        connections = [[] for _ in contours]
        for i, a in enumerate(contours):
            a_com = a.mean(axis=0)[0]
            a_area = self.contourAreas[i]
            for j, b in enumerate(contours[i + 1:]):
                max_metric = max(a_area, self.contourAreas[j]) * METRIC_SCALE_FACTOR
                b_com = b.mean(axis=0)[0]
                d = a_com - b_com
                metric = d[0] ** 2 + d[1] ** 2
                if (metric < max_metric):
                    connections[i].append(j + i + 1)
                    connections[j + i + 1].append(i)
        # Breadth first search from each contour that wasn't yet assigned to a group to find its group
        assigned = [False for _ in contours]
        group = set()
        groups = []
        for i, c in enumerate(connections):
            if assigned[i]: continue
            group = {i}
            assigned[i] = True
            horizon = set(c)
            while (True):
                if len(horizon) == 0:
                    # Group completed
                    groups.append(list(group))
                    break
                for i in horizon: assigned[i] = True
                group.update(horizon)
                new_horizon = set()
                for old in horizon:
                    new_horizon.update(connections[old])
                new_horizon.difference_update(group)
                horizon = new_horizon
        return groups

    def rank_groups(self, contours: np.ndarray, groups: List[List[int]]) -> Optional[List[int]]:
        """Returns the group that is most likely to be the target
        Could use metrics such as curvature, consistency of contours, consistency with last frame

        Currently just returns the group with most elements
        """
        return max((g for g in groups if len(g) > 2), key=lambda x: sum(self.contourAreas[i] for i in x), default=None)

    def get_values(self, contours: np.ndarray, group: List[int], frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """Returns position of target in normalized device coordinates from contour group
        """
        # Calculates mean of contour centers weighted by their areas
        summed = np.array((0.0, 0.0))
        total_area = 0
        for c in group:
            area = self.contourAreas[c]
            summed += contours[c].mean(axis=0)[0] * area
            total_area += area
        weighted_position =  summed / total_area
        cv2.circle(self.frame, tuple(map(int, weighted_position)), 5, (0, 0, 255), -1)
        return (
            weighted_position[0] * 2.0 / frame_size[0] - 1.0,
            weighted_position[1] * 2.0 / frame_size[1] - 1.0,
        )


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py

    vision = Vision(
        CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
        NTConnection(),
    )
    while True:
        vision.run()
