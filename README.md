# ROSBAG Vehicle Removal

## Installation
This code uses [PointRCNN](https://github.com/sshaoshuai/PointRCNN.git) as the backend.
Additional training might be required depending on data to be cleaned.

The input and output are  `rosbag` file format.
Therefore, please install ROS can convert your required data to a lidar message in ROS format.

## RUN
First, run `roscore` and setup required `rosparam` values (a list of ros parameters used are mentioned inside `rosparam_()` function in the `ros_veh_remove.py` file).
Next, simply run `python ros_veh_remove.py` to clean the input rosbag files and place them separately in an output directory.
