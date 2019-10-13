# opencv-mono-detector
Simple object detection (by color) using mono-camera and reprojection into the 3D ray

Features:
 - Dectect object with specific color (HSV)
 - Reproject object position to the 3D ray (calibrated camera needed)
 - Estimate object position in 3D when object size is known
 - Works with and without ROS
 - Helper launch file to simplify calibration
 
## Run
### 1. Calibrate camera
You can use provided .launch-files, change them if you calibration pattern or camera settings are different

### 2.Configure object detection parameters
2.1. Run script with full GUI
```bash
./detector.py --calibration <path_to_calibration> --config <path_to_new_config> --gui full
```
2.2. Setup color threshold, blur, size threshold to detect object
2.3. Press "s" to save the config

## 3. Run with ROS
3.1. Run `roscore`
```
roscore
```
3.2. Run the script
```bash
./detector.py --calibration <path_to_calibration> --config <path_to_new_config> --gui base --ros
```
3.3. Open RViz
```
rviz
```
3.4. Open RViz config `cfg/config.rviz` to see estimated object position


## Usage
```
usage: detector.py [-h] [--capture CAPTURE] [--config CONFIG]
                   [--calibration CALIBRATION] [--size SIZE]
                   [--gui {base,full}]

Find object with given color and calculate ray or position

optional arguments:
  -h, --help            show this help message and exit
  --capture CAPTURE     OpenCV video capture
  --config CONFIG       Path to the configuration file to open/save
  --calibration CALIBRATION
                        Path to the camera calibration file
  --size SIZE           Object size (diameter) in meters
  --gui {base,full}     Show GUI
  --ros                 Enable ROS
```
