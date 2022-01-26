from camera_manager import CameraManager
from connection import NTConnection
from magic_numbers import ERODE_DILATE_KERNEL, METRIC_SCALE_FACTOR, TARGET_HSV_HIGH, TARGET_HSV_LOW
from typing import Tuple, Optional, List
import cv2
import numpy as np
import time


class Vision:
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection

    def run(self) -> None:
        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """

        frame_time, frame = self.camera_manager.get_frame()
        # frame time is 0 in case of an error
        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return

        # Flip the image beacuse it's originally upside down.
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        results, display = process_image(frame)

        if results is not None:
            x, y = results
            self.connection.send_results(
                (x, y, time.monotonic())
            )  # x and y in NDC, positive axes right and down; timestamp

        # send image to display on driverstation
        self.camera_manager.send_frame(display)
        self.connection.set_fps()


def process_image(
    frame: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """Takes a frame returns the target dist & angle and an annotated display
    returns None if there is no target
    """

    mask = preprocess(frame)
    contours = find_contours(mask)
    contour_areas = [cv2.contourArea(c) for c in contours]
    groups = group_contours(contours, contour_areas)
    best_group = rank_groups(groups, contour_areas)
    if best_group is None:
        display = annotate_image(frame, contours, [], (-1, -1))
        return (None, display)
    values = get_values(
        contours, best_group, (frame.shape[1], frame.shape[0]), contour_areas
    )
    display = annotate_image(frame, contours, best_group, values)
    return values, display


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Creates a mask of expected target green color"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TARGET_HSV_LOW, TARGET_HSV_HIGH)
    mask = cv2.erode(mask, ERODE_DILATE_KERNEL)
    mask = cv2.dilate(mask, ERODE_DILATE_KERNEL)
    return mask


def find_contours(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds contours on a grayscale mask"""
    *_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def group_contours(contours: np.ndarray, contour_areas: List[int]) -> List[List[int]]:
    """Returs a nested list of contour indices grouped by relative distance"""
    # Build an adjacency list based on distance between contours relative to their area
    connections = [[] for _ in contours]
    for i, a in enumerate(contours):
        a_com = a.mean(axis=0)[0]
        a_area = contour_areas[i]
        for j, b in enumerate(contours[i + 1 :]):
            max_metric = max(a_area, contour_areas[j + i + 1]) * METRIC_SCALE_FACTOR
            b_com = b.mean(axis=0)[0]
            d = a_com - b_com
            metric = d[0] ** 2 + d[1] ** 2
            if metric < max_metric:
                connections[i].append(j + i + 1)
                connections[j + i + 1].append(i)
    # Breadth first search from each contour that wasn't yet assigned to a group to find its group
    assigned = [False for _ in contours]
    group = set()
    groups = []
    for i, c in enumerate(connections):
        if assigned[i]:
            continue
        group = {i}
        assigned[i] = True
        horizon = set(c)
        while True:
            if len(horizon) == 0:
                # Group completed
                groups.append(list(group))
                break
            for i in horizon:
                assigned[i] = True
            group.update(horizon)
            new_horizon = set()
            for old in horizon:
                new_horizon.update(connections[old])
            new_horizon.difference_update(group)
            horizon = new_horizon
    return groups


def rank_groups(
    groups: List[List[int]], contour_areas: np.ndarray
) -> Optional[List[int]]:
    """Returns the group that is most likely to be the target
    Takes the group with the largest combined area that has >1 contour
    """
    # throw away groups with only 1 target as they are likely a false positive
    valid_groups = (g for g in groups if len(g) > 1)
    return max(
        valid_groups,
        key=lambda x: sum(contour_areas[i] for i in x),
        default=None,
    )


def get_values(
    contours: np.ndarray,
    group: List[int],
    frame_size: Tuple[int, int],
    contour_areas: List[int],
) -> Tuple[float, float]:
    """Returns position of target in normalized device coordinates from contour group"""
    # Calculates mean of contour centers weighted by their areas
    summed = np.array((0.0, 0.0))
    total_area = 0
    for c in group:
        area = contour_areas[c]
        summed += contours[c].mean(axis=0)[0] * area
        total_area += area
    weighted_position = summed / total_area

    return (
        weighted_position[0] * 2.0 / frame_size[0] - 1.0,
        weighted_position[1] * 2.0 / frame_size[1] - 1.0,
    )


def annotate_image(
    display: np.ndarray, contours: np.ndarray, group: List[int], pos: Tuple[int, int]
) -> np.ndarray:
    cv2.drawContours(
        display,
        contours,
        -1,
        (255, 0, 0),
        thickness=2,
    )
    x = int((pos[0] + 1) / 2 * display.shape[1])
    y = int((pos[1] + 1) / 2 * display.shape[0])

    for c1, c2 in zip(group, group[1:]):
        # takes the first point in each contour to be fast
        p1 = contours[c1][0][0]  # each point is [[x, y]]
        p2 = contours[c2][0][0]
        cv2.line(display, p1, p2, (0, 255, 0), 1)

    cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
    return display


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py

    vision = Vision(
        CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
        NTConnection(),
    )
    while True:
        vision.run()
