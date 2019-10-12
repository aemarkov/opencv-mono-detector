#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import yaml
import os

from settings import Settings
import ui

def init_argparse():
    parser = argparse.ArgumentParser(description='Find object with given color and calculate ray or position')
    parser.add_argument('--capture', type=int, default=0, help='OpenCV video capture')
    parser.add_argument('--config', help='Path to the configuration file to open/save')
    parser.add_argument('--calibration', help='Path to the camera calibration file')
    parser.add_argument('--size', type=float, help='Object size (diameter) in meters')
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

# Read matrix with given shape from file
def parse_array(node):
    return np.reshape(np.array(node['data']), (node['rows'], node['cols']))

# Load calibration data from file
def load_calib(path):
    with open(path, 'r') as f:
        calib = yaml.load(f)
        width = calib['image_width']
        height = calib['image_height']
        camera_matrix = parse_array(calib['camera_matrix'])
        distortion = parse_array(calib['distortion_coefficients'])
        return (width, height), camera_matrix, distortion

# Init matrices to perform undistortion using remap()
def init_undistort(size, camera_mat, dist_coefs, alpha):
    new_camera_mat, _ = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coefs, size, alpha)
    return cv2.initUndistortRectifyMap(camera_mat, dist_coefs, None, new_camera_mat, size, cv2.CV_16SC2)

# Binarize and find object
def find_object(frame, settings):
    # Gaussian blur kernel size should be odd
    if settings.blur.value % 2 == 0:
        settings.blur.value += 1

    # Binarize image to find color
    blurred = cv2.GaussianBlur(frame, (settings.blur.value, settings.blur.value), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (settings.h_min.value, settings.s_min.value, settings.v_min.value),
                              (settings.h_max.value, settings.s_max.value, settings.v_max.value))

    cv2.imshow('binary', binary)

    # Find all contours
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Find contour with biggest area
    index, area = max([(index, cv2.contourArea(contour)) for index, contour in enumerate(contours)], key=lambda x: x[1])
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    if area < settings.min_size.value:
        return None

    # Find biggest contour center
    contour = contours[index]
    moments = cv2.moments(contour)
    center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

    cv2.drawContours(frame, contours, index, (0, 255, 0), 3)
    cv2.circle(frame, center, 5, (0, 255, 255), 3)

    return center


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
    ui.create('controls', config)

    size, camera_mat, dist_coefs = load_calib(args.calibration)
    mat1, mat2 = init_undistort(size, camera_mat, dist_coefs, 1.0)

    # Read video from camera and process
    cap = cv2.VideoCapture(args.capture)
    while True:
        ui.read('controls', config)

        # Read and undistort image from camera
        _, frame = cap.read()
        undistorted = cv2.remap(frame, mat1, mat2, cv2.INTER_LINEAR)

        center = find_object(undistorted, config)

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