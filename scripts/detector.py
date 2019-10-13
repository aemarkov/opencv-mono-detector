#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import yaml
import os

from settings import Settings
from calib import CameraCalibration
import ui

try:
    import rospy
    from std_msgs.msg import ColorRGBA
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3
    from visualization_msgs.msg import Marker
    is_ros = True
except ImportError:
    print('Failed to import ROS')
    is_false = True

def init_argparse():
    parser = argparse.ArgumentParser(description='Find object with given color and calculate ray or position')
    parser.add_argument('--capture', type=int, default=0, help='OpenCV video capture')
    parser.add_argument('--config', help='Path to the configuration file to open/save')
    parser.add_argument('--calibration', help='Path to the camera calibration file')
    parser.add_argument('--size', type=float, help='Object size (diameter) in meters')
    parser.add_argument('--gui', choices=['base', 'full'], help='Show GUI')
    return parser


def make_default_config():
    return Settings(
        h_min=Settings(i=0, value=0, max=180),
        h_max=Settings(i=1, value=180, max=180),
        s_min=Settings(i=2, value=0, max=255),
        s_max=Settings(i=3, value=255, max=255),
        v_min=Settings(i=4, value=0, max=255),
        v_max=Settings(i=5, value=255, max=255),
        blur=Settings(i=6, value=0, max=21),
        min_size=Settings(i=7, value=0, max=5000)
    )

# Init matrices to perform undistortion using remap()
def init_undistort(calib, alpha):
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(calib.camera_matrix, calib.distortion, calib.size, alpha)
    return cv2.initUndistortRectifyMap(calib.camera_matrix, calib.distortion, None, new_camera_matrix, calib.size, cv2.CV_16SC2)

# Binarize and find object
def find_object(frame, settings, args):
    # Gaussian blur kernel size should be odd
    if settings.blur.value % 2 == 0:
        settings.blur.value += 1

    # Binarize image to find color
    blurred = cv2.GaussianBlur(frame, (settings.blur.value, settings.blur.value), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (settings.h_min.value, settings.s_min.value, settings.v_min.value),
                              (settings.h_max.value, settings.s_max.value, settings.v_max.value))

    if args.gui == 'full':
        cv2.imshow('binary', binary)

    # Find all contours
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Find contour with biggest area
    index, area = max([(index, cv2.contourArea(contour)) for index, contour in enumerate(contours)], key=lambda x: x[1])

    if args.gui == 'full':
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    if area < settings.min_size.value:
        return None

    # Find biggest contour center
    contour = contours[index]
    moments = cv2.moments(contour)
    center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

    if args.gui == 'base' or args.gui == 'full':
        cv2.drawContours(frame, contours, index, (0, 255, 0), 3)
        cv2.circle(frame, center, 5, (0, 255, 255), 3)

    return center

def reproject(center, calib):
    u, v = center
    Cx = calib.camera_matrix[0, 2]
    Cy = calib.camera_matrix[1, 2]
    fx = calib.camera_matrix[0, 0]
    fy = calib.camera_matrix[1, 1]
    return np.array([(u - Cx)/fx, (v - Cy)/fy, 1])

def publish(vector, pose_pub, line_pub):
    x, y, z = vector
    point = Point(z, x, y)

    pose = PoseStamped()
    pose.header.frame_id = 'map' # TODO: Use camera frame
    pose.pose.position = point

    marker = Marker()
    marker.type = Marker.LINE_LIST
    marker.header.frame_id = 'map'
    marker.scale = Vector3(0.01, 0, 0)
    marker.color = ColorRGBA(1, 0, 0, 1)
    marker.points.append(Point(0, 0, 0))
    marker.points.append(point)

    pose_pub.publish(pose)
    line_pub.publish(marker)

def main():
    parser = init_argparse()
    args = parser.parse_args()

    print('----------------------------------------------')
    print('OpenCV object detector')
    print('Controls:')
    print('    - S     Save current config')
    print('    - ESC   Exit without saving current config')
    print('----------------------------------------------')

    config = Settings.load(args.config, default=make_default_config())

    if args.gui == 'full':
        ui.create('controls', config)

    calibration = CameraCalibration.load(args.calibration)
    mat1, mat2 = init_undistort(calibration, 1.0)

    if is_ros:
        rospy.init_node('object_finder')
        pose_pub = rospy.Publisher('object', PoseStamped, queue_size=1)
        line_pub = rospy.Publisher('line', Marker, queue_size=1)

    # Read video from camera and process
    cap = cv2.VideoCapture(args.capture)
    while True:
        if args.gui == 'full':
            ui.read('controls', config)

        # Read and undistort image from camera
        _, frame = cap.read()
        undistorted = cv2.remap(frame, mat1, mat2, cv2.INTER_LINEAR)

        center = find_object(undistorted, config, args)
        if center != None:
            point_3d = reproject(center, calibration)
            if is_ros:
                publish(point_3d, pose_pub, line_pub)

        if args.gui == 'base' or args.gui == 'full':
            cv2.imshow('rgb', undistorted)

        key = cv2.waitKey(1)
        if key == 115:
            print('Saving...')
            config.store(args.config)
        if key == 27:
            print('Exit')
            exit(0)

if __name__ == "__main__":
    main()