import numpy as np

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
TARGET_HSV_LOW = np.array((80, 180, 60), dtype=np.uint8)
TARGET_HSV_HIGH = np.array((110, 255, 255), dtype=np.uint8)

# bad for testing
# TARGET_HSV_LOW = np.array((30, 40, 30), dtype=np.uint8)
# TARGET_HSV_HIGH = np.array((150, 255, 255), dtype=np.uint8)

# Mask erode/dilate kernel
ERODE_DILATE_KERNEL = np.ones((3, 3), dtype=np.float32)

# Factor by which areas of contours are multiplied to get an estimate for allowed distance squared between them
METRIC_SCALE_FACTOR = 27.5
