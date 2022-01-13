# vision-2022
The Drop Bears' vision code for FIRST Rapid React (FRC 2022)

## Setup
To install dependencies do:

    pip install -r requirments.txt

# Run
## Desktop
Use `test_images.py` to run vision code on test images and see any that are wrong

    pytest test_images.py

Use `sim.py` to view the results for a single image, video, or webcam on your computer
e.g.

    python3 sim.py -i test_images/other/NearLaunchpad8ft10in.png 

## Deploying
You can either use `deploy.py` or the web console

For both of these you must first connect to the robots wifi network

for more info about the web console can be found [here](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/the-raspberry-pi-frc-console.html)

deploy.py should work with just

    python3 deploy.py