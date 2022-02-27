import csv
import pytest
import cv2
import vision


def read_test_data_csv(fname: str):
    with open(fname) as f:
        result = []
        for image, x, y in csv.reader(f):
            result.append((image, float(x), float(y)))
    return result


images = read_test_data_csv("test_images/expected.csv")
fail_images = read_test_data_csv("test_images/expected_fail.csv")

# TODO: could calculate the allowed error based on what would result in a shot missing
allowed_x_error = 0.1
allowed_y_error = 0.1


@pytest.mark.parametrize("filename,expected_x,expected_y", images)
def test_sample_images(filename: str, expected_x: float, expected_y: float):
    image = cv2.imread(f"./test_images/other/{filename}")
    assert image is not None
    results, _ = vision.process_image(image)

    assert results is not None
    output_x, output_y, _ = results

    x_error = abs(output_x - expected_x)
    y_error = abs(output_y - expected_y)

    assert x_error < allowed_x_error
    assert y_error < allowed_y_error


@pytest.mark.parametrize("filename,expected_x,expected_y", fail_images)
@pytest.mark.xfail
def test_sample_images_fail(filename: str, expected_x: float, expected_y: float):
    image = cv2.imread(f"./test_images/other/{filename}")
    assert image is not None
    results, _ = vision.process_image(image)

    assert results is not None
    output_x, output_y, _ = results

    x_error = abs(output_x - expected_x)
    y_error = abs(output_y - expected_y)

    assert x_error < allowed_x_error
    assert y_error < allowed_y_error
