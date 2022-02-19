from camera_manager import CameraManager
from connection import NTConnection
from magic_numbers import (
    ERODE_DILATE_KERNEL,
    METRIC_SCALE_FACTOR,
    TARGET_HSV_HIGH,
    TARGET_HSV_LOW,
)
from typing import Tuple, Optional, List
from math import tan
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
        self.connection.pong()
        frame_time, frame = self.camera_manager.get_frame()
        # frame time is 0 in case of an error
        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return

        # Flip the image beacuse it's originally upside down.
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        results, display = process_image(frame)

        if results is not None:
            angle, distance, confidence = results
            self.connection.send_results(
                (angle, distance, confidence, time.monotonic())
            )  # x and y in NDC, positive axes right and down; timestamp

        # send image to display on driverstation
        self.camera_manager.send_frame(display)
        self.connection.set_fps()


def process_image(
    frame: np.ndarray,
) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray]:
    """Takes a frame returns the target dist & angle and an annotated display
    returns None if there is no target
    """

    mask = preprocess(frame)
    contours = filter_contours(find_contours(mask))
    contour_areas = [cv2.contourArea(c) for c in contours]
    groups = group_contours(contours, contour_areas)
    best_group = rank_groups(groups, contours, contour_areas)
    if best_group is None:
        display = annotate_image(frame, contours, [], (-1, -1))
        cv2.circle(display, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), 5, (0, 255, 255), -1)
        return (None, display)
    pos = group_com(contours, best_group, contour_areas)
    display = annotate_image(frame, contours, best_group, pos)

    norm_x = pos[0] * 2.0 / FRAME_WIDTH - 1.0
    angle = norm_x * MAX_FOV_WIDTH / 2
    # Trigonometrically estimated from the group's COM height on the screen
    vert_angle = GROUND_ANGLE - (pos[1] * 2.0 / FRAME_HEIGHT - 1.0) * MAX_FOV_HEIGHT / 2
    distance = REL_TARGET_HEIGHT / tan(vert_angle) + DISTANCE_CORRECTION
    conf = group_confidence(best_group, contours, contour_areas) * (1.0 - abs(norm_x))
    # print(distance, conf)

    return (angle, distance, conf), display


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


def filter_contours(contours: List[np.ndarray]) -> List[np.ndarray]:
    """Filters contours based on their aspect ratio, discaring tall ones"""

    def is_contour_good(contour: np.ndarray):
        _, _, w, h = cv2.boundingRect(contour)
        return w / h > MIN_CONTOUR_ASPECT_RATIO

    return [c for c in contours if is_contour_good(c)]


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
    groups: List[List[int]], contours: np.ndarray, contour_areas: np.ndarray
) -> Optional[List[int]]:
    """Returns the group that is most likely to be the target
    Takes the group with the largest combined area that has >1 contour
    """
    # throw away groups with only 1 target as they are likely a false positive
    valid_groups = (g for g in groups if len(g) > 1)
    return max(
        valid_groups,
        key=lambda x: group_fitness(x, contours, contour_areas),
        default=None,
    )


def lerp(
    value: float,
    input_lower: float,
    input_upper: float,
    output_lower: float,
    output_upper: float,
) -> float:
    """Scales a value based on the input range and output range.
    For example, to scale a joystick throttle (1 to -1) to 0-1, we would:
        scale_value(joystick.getThrottle(), 1, -1, 0, 1)
    """
    input_distance = input_upper - input_lower
    output_distance = output_upper - output_lower
    ratio = (value - input_lower) / input_distance
    result = ratio * output_distance + output_lower
    return result


def group_fitness(
    group: List[int],
    contours: np.ndarray,
    contour_areas: np.ndarray,
) -> float:
    """Fittness function for ranking the groups, unitless"""
    bounding_rects = [cv2.boundingRect(contours[i]) for i in group]
    min_y = min(rect[1] for rect in bounding_rects)
    max_y = max(rect[1] + rect[3] for rect in bounding_rects)
    height = max_y - min_y
    return sum(contour_areas[i] for i in group) * TOTAL_AREA_K + height**2 * HEIGHT_K


def group_confidence(
    group: List[int], contours: np.ndarray, contour_areas: np.ndarray
) -> float:
    """Confidence for a group 0-1"""
    # work out aspect ratio
    bounding_rects = [cv2.boundingRect(contours[i]) for i in group]
    min_x = min(rect[0] for rect in bounding_rects)
    max_x = max(rect[0] + rect[2] for rect in bounding_rects)
    min_y = min(rect[1] for rect in bounding_rects)
    max_y = max(rect[1] + rect[3] for rect in bounding_rects)
    height = max_y - min_y
    width = max_x - min_x
    aspect_ratio = width / height
    aspect_ratio_error = abs(1 - GROUP_ASPECT_RATIO / aspect_ratio)

    # how close to being a rectangle each contour is
    rects_area = sum(rect[2] * rect[3] for rect in bounding_rects)
    real_area = sum(contour_areas[i] for i in group)
    rectangulares = real_area / rects_area

    length = lerp(len(group), 2, 5, 0.5, 1)

    return (
        lerp(aspect_ratio_error, 0, SATURATING_ASPECT_RATIO_ERROR, 1, 0)
        * CONF_ASPECT_RATIO_K
        + rectangulares * CONF_RECTANGULARES_K
        + length * CONF_LENGTH_K
    ) / CONF_TOTAL


def group_com(
    contours: np.ndarray,
    group: List[int],
    contour_areas: List[int],
) -> Tuple[int, int]:
    """Return center of mass of a contour group"""
    # Calculates mean of contour centers weighted by their areas
    summed = np.array((0.0, 0.0))
    total_area = 0
    for c in group:
        area = contour_areas[c]
        summed += contours[c].mean(axis=0)[0] * area
        total_area += area
    weighted_position = summed / total_area  # xy position
    return (
        int(weighted_position[0]),
        min(min(p[0][1] for p in contours[c]) for c in group),
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
    for c1, c2 in zip(group, group[1:]):
        # takes the first point in each contour to be fast
        p1 = contours[c1][0][0]  # each point is [[x, y]]
        p2 = contours[c2][0][0]
        cv2.line(display, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 1)

    cv2.circle(display, pos, 5, (0, 0, 255), -1)
    return display


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py

    vision = Vision(
        CameraManager(
            "Power Port Camera", "/dev/video0", FRAME_HEIGHT, FRAME_WIDTH, 30, "kYUYV"
        ),
        NTConnection(),
    )
    while True:
        vision.run()
