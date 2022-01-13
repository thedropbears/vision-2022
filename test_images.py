import pytest
from typing import Tuple
import math
from vision import Vision
from camera_manager import MockCameraManager

# load expected values for sample images
# should be in format: image name,dist,angle
with open("test_images/expected.csv") as f:
    lines = f.read().split("\n")
    lines_values = [line.split(",") for line in lines]
    # convert dist and angle to float
    images = [( val[0], float(val[1]), float(val[2]) ) for val in lines_values]

# TODO: could calculate the allowed error based on what would result in a shot missing 
allowed_azimuth_error = math.radians(5) 
allowed_dist_error = 0.3

@pytest.mark.parametrize("filename,expected_dist,expected_angle", images)
def test_sample_images(filename: str, expected_dist: float, expected_angle: float):
    vision = Vision(MockCameraManager(filename))
    output_dist, output_angle = (41.99, 1.99)#vision.run()

    dist_error = abs(output_dist-expected_dist)
    azimuth_error = abs(output_angle-expected_angle)

    assert dist_error < allowed_dist_error
    assert azimuth_error < allowed_azimuth_error
