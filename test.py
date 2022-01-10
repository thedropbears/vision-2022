from vision import Vision
from camera_manager import MockCameraManager

with open("test_images/expected.csv") as f:
    lines = f.read().split("\n")
    images = [line.split(",") for line in lines]

print(f"{len(images)} images")

avg_error = [0 for _ in images[0][1:]]

for image in images:
    vision = Vision(MockCameraManager(image))
    # convert expected outputs to float
    expected_outputs = map(float, image[1:])
    # get real outputs
    outputs = (50, 1.5)# vision.run()
    # calcualte error
    error = [(output-expected_output)/output for output, expected_output in zip(outputs, expected_outputs)]
    print(f"{image[0]}: {', '.join([str(e)+'%' for e in error])}")
    # add to average error
    for idx, val in enumerate(error):
        avg_error[idx] += val

print(f"\navg error: {', '.join([str(e/len(images))+'%' for e in error])}")
