# opencv-mono-detector
Simple object detection (by color) using mono-camera and reprojection into the 3D ray for ROS

Features:
 - Dectect object with specific color (HSV)
 - Reproject object position to the 3D ray (calibrated camera needed)
 - Estimate object position in 3D when object size is known
 - Helper launch file to simplify calibration
 
## Run
### 1. Calibrate camera
You can use provided .launch-files, change them if you calibration pattern or camera settings are different

## 2. Run
2.1. Run `roscore`
```
roscore
```
2.2. Run your camera
2.3. Run in the configuration mode to setup detection paraneters
```
rosrun opencv-mono-detector detector.py --config <path_to_new_config> --gui full
```
Setup color threshold, blur, size threshold to detect object

Press "s" to save the config

2.2. Run in "production" mode
```bash
rosrun opencv-mono-detector detector.py --config <path_to_new_config>
```
2.3. Open RViz
```
rviz
```
2.4. Open RViz config `cfg/config.rviz` to see estimated object position


## Usage
```
usage: detector.py [-h][--config CONFIG]
                   [--calibration CALIBRATION] [--size SIZE]
                   [--gui {base,full}]

Find object with given color and calculate ray or position

optional arguments:
  -h, --help            show this help message and exit
  --capture CAPTURE     OpenCV video capture
  --config CONFIG       Path to the configuration file to open/save
  --size SIZE           Object size (diameter) in meters
  --gui {base,full}     Show GUI
```
