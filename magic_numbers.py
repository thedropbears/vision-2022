import numpy as np
# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
TARGET_HSV_LOW = np.array((60, 180, 80), dtype=np.uint8)
TARGET_HSV_HIGH = np.array((100, 255, 255), dtype=np.uint8)

# Mask erode/dilate kernel
ERODE_DILATE_KERNEL = np.ones((3, 3), dtype=np.uint0)

# Factor by which areas of contours are multiplied to get an estimate for allowed distance squared between them
METRIC_SCALE_FACTOR = 27.5
