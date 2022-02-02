import numpy as np
import math

# Camera settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Target HSV for masking
TARGET_HSV_LOW = np.array((50, 150, 60), dtype=np.uint8)
TARGET_HSV_HIGH = np.array((110, 255, 255), dtype=np.uint8)

# bad for testing
# TARGET_HSV_LOW = np.array((30, 40, 30), dtype=np.uint8)
# TARGET_HSV_HIGH = np.array((150, 255, 255), dtype=np.uint8)

# Mask erode/dilate kernel
ERODE_DILATE_KERNEL = np.ones((3, 3), dtype=np.float32)

MIN_ASPECT_RATIO = 0.9

# Factor by which areas of contours are multiplied to get an estimate for allowed distance squared between them
METRIC_SCALE_FACTOR = 26

# The following were careful measurements of the frame area with the camera
# aimed at a flat wall, and the distance of the camera from the wall. All are in
# millimetres.
FOV_WIDTH = 1793
FOV_HEIGHT = 2303
FOV_DISTANCE = 2234

MAX_FOV_WIDTH = math.atan2(FOV_WIDTH / 2, FOV_DISTANCE) * 2  # 54.54 degrees
MAX_FOV_HEIGHT = math.atan2(FOV_HEIGHT / 2, FOV_DISTANCE) * 2  # 42.31 degrees

# Angle which camera's main axis forms with the horizontal plane
GROUND_ANGLE = 0.1
TARGET_HEIGHT = 2.64
ROBOT_HEIGHT = 1
REL_TARGET_HEIGHT = TARGET_HEIGHT - ROBOT_HEIGHT
RAW_AREA_C = 100
# Weightings for angle- and area- based distance estimations
TRIG_DISTANCE_K = 0.5
AREA_DISTANCE_K = 1 - TRIG_DISTANCE_K

# Coeffecients for group fitness
HEIGHT_K = -4
TOTAL_AREA_K = 1
