import numpy as np
import math

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
TARGET_HSV_LOW = np.array((50, 150, 60), dtype=np.uint8)
TARGET_HSV_HIGH = np.array((110, 255, 255), dtype=np.uint8)

# bad for testing
# TARGET_HSV_LOW = np.array((30, 40, 30), dtype=np.uint8)
# TARGET_HSV_HIGH = np.array((150, 255, 255), dtype=np.uint8)

# Mask erode/dilate kernel
ERODE_DILATE_KERNEL = np.ones((3, 3), dtype=np.float32)

# Factor by which areas of contours are multiplied to get an estimate for allowed distance squared between them
METRIC_SCALE_FACTOR = 25

# The following were careful measurements of the frame area with the camera
# aimed at a flat wall, and the distance of the camera from the wall. All are in
# millimetres.
FOV_WIDTH = 1793
FOV_HEIGHT = 2303
FOV_DISTANCE = 2234

MAX_FOV_WIDTH = math.atan2(FOV_WIDTH / 2, FOV_DISTANCE) * 2  # 54.54 degrees
MAX_FOV_HEIGHT = math.atan2(FOV_HEIGHT / 2, FOV_DISTANCE) * 2  # 42.31 degrees
