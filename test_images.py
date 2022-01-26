import pytest
import cv2
from connection import DummyConnection
import vision
from camera_manager import MockImageManager

# load expected values for sample images
# should be in format: image name,dist,angle
with open("test_images/expected.csv") as f:
    lines = f.read().split("\n")
    lines_values = [line.split(",") for line in lines]
    print(lines_values)
    # convert dist and angle to float
    images = [
        (val[0], float(val[1]), float(val[2])) for val in lines_values if len(val) == 3
    ]

# TODO: could calculate the allowed error based on what would result in a shot missing
allowed_x_error = 0.1
allowed_y_error = 0.1


@pytest.mark.parametrize("filename,expected_x,expected_y", images)
def test_sample_images(filename: str, expected_x: float, expected_y: float):
    image = cv2.imread(f"./test_images/other/{filename}")
    assert not image is None
    results, _ = vision.process_image(image)

    assert results is not None
    output_x, output_y = results

    x_error = abs(output_x - expected_x)
    y_error = abs(output_y - expected_y)

    assert x_error < allowed_x_error
    assert y_error < allowed_y_error
