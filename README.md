# Map Cleaning


## Paper

Code release for the paper titled **"Semantic Detection and Custom Tracking based 3D Lidar Map Cleaning Approach for Autonomous Vehicle Localization"**, IROS 2020.

## Installation
This code uses [PointRCNN](https://github.com/sshaoshuai/PointRCNN.git) as the backend.
Additional training might be required depending on data to be cleaned.

The input and output are  `rosbag` file format.
Therefore, please install ROS can convert your required data to a lidar message in ROS format.

## RUN
First, run `roscore` and setup required `rosparam` values (a list of ros parameters used are mentioned inside `rosparam_()` function in the `map_clean.py` file).
Next, simply run `python map_clean.py` to clean the input rosbag files and place them separately in an output directory.

## TODO
- Add `map_clean.py`
- Add a gif describing our work
