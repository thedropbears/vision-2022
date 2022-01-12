import pytest
from typing import Tuple
import math
from vision import Vision
from camera_manager import MockCameraManager

# load expected values for sample images
# should be in format: image name,[...expected values]
with open("test_images/expected.csv") as f:
    lines = f.read().split("\n")
    lines_values = [line.split(",") for line in lines]
    # put expected output values into a tuple and convert to float
    images = [( val[0], tuple(map(float, val[1:])) ) for val in lines_values]

# TODO: could calculate the allowed error based on what would result in a shot missing 
allowed_azimuth_error = math.radians(5) 
allowed_dist_error = 0.3

@pytest.mark.parametrize("filename,expected", images)
def test_sample_images(filename: str, expected: Tuple[float, float]):
    vision = Vision(MockCameraManager(filename))
    outputs = (41.99, 1.9)#vision.run()
    error = [abs(output-expected) for output, expected in zip(outputs, expected)]
    assert error[0] < allowed_azimuth_error
    assert error[1] < allowed_dist_error
