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

# Convert dictionary to the python object for more convenient usage
def dict_to_obj(name, dictionary):
    obj =  type(name, (object,), {})()
    for key in dictionary:
        if type(dictionary[key]) is dict:
            obj.__dict__[key] = dict_to_obj(key.title(), dictionary[key])
        else:
            obj.__dict__[key] = dictionary[key]
    return obj

def make_default_config():
    return {
        'color_threshold': {
            'min': { 'h': 0, 's': 0, 'v': 0 },
            'max': { 'h': 0, 's': 0, 'v': 0}
        },
        'blur': 0,
        'area_threshold': 0
    }

# Load settings from YAML file
def load_settings(file):
    if file is None or not os.path.isfile(file):
        return dict_to_obj(make_default_config())

    with open(file, 'r') as f:
        return dict_to_obj(yaml.load(f.read()))


def none(a):
    pass

# Create trackbars
def create_ui(settings):
    cv2.namedWindow('controls')
    cv2.createTrackbar('H min', 'controls', settings['color_threshold']['min']['h'], 180, none)
    cv2.createTrackbar('H max', 'controls', settings['color_threshold']['max']['h'], 180, none)
    cv2.createTrackbar('S min', 'controls', settings['color_threshold']['min']['s'], 255, none)
    cv2.createTrackbar('S max', 'controls', settings['color_threshold']['max']['s'], 255, none)
    cv2.createTrackbar('V min', 'controls', settings['color_threshold']['min']['v'], 255, none)
    cv2.createTrackbar('V max', 'controls', settings['color_threshold']['min']['v'], 255, none)
    cv2.createTrackbar('blur',  'controls', settings['blur'], 21, none)
    cv2.createTrackbar('threshold', 'controls', settings['area_threshold'], 5000, none)

# Get current trackbars values
def update_settings(settings):
    settings['color_threshold']['min']['h'] = cv2.getTrackbarPos('H min', 'controls')
    settings['color_threshold']['max']['h'] = cv2.getTrackbarPos('H max', 'controls')
    settings['color_threshold']['min']['s'] = cv2.getTrackbarPos('S min', 'controls')
    settings['color_threshold']['max']['s'] = cv2.getTrackbarPos('S max', 'controls')
    settings['color_threshold']['min']['v'] = cv2.getTrackbarPos('V min', 'controls')
    settings['color_threshold']['max']['v'] = cv2.getTrackbarPos('V max', 'controls')
    settings['blur'] = cv2.getTrackbarPos('blur', 'controls')
    settings['area_threshold']['min']['h'] = cv2.getTrackbarPos('threshold', 'controls')
    if settings['blur'] % 2 != 1:
        settings['blur'] += 1

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
    # Binarize image to find color
    blurred = cv2.GaussianBlur(frame, (settings['blur'], settings['blur']), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (settings['color']['min']['h'], settings['color']['min']['s'], settings['color']['min']['v']),
                              (settings['color']['max']['h'], settings['color']['max']['s'], settings['color']['max']['v']))

    # Find all contours
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Find contour with biggest area
    index, area = max([(index, cv2.contourArea(contour)) for index, contour in enumerate(contours)], key=lambda x: x[1])
    #if settings.args.is_gui:
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    if area < settings['area_threshold']:
        return None

    # Find biggest contour center
    contour = contours[index]
    moments = cv2.moments(contour)
    center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
    #if settings.args.is_gui:
    cv2.drawContours(frame, contours, index, (0, 255, 0), 3)
    cv2.circle(frame, center, 5, (0, 255, 255), 3)

    return center


def main():
    parser = init_argparse()
    args = parser.parse_args()

    settings = load_settings(args.config)
    create_ui(settings)

    size, camera_mat, dist_coefs = load_calib(args.calibration)
    mat1, mat2 = init_undistort(size, camera_mat, dist_coefs, 1.0)

    # Read video from camera and process
    cap = cv2.VideoCapture(args.capture)
    while True:
        update_settings(settings)

        # Read and undistort image from camera
        _, frame = cap.read()
        undistorted = cv2.remap(frame, mat1, mat2, cv2.INTER_LINEAR)

        center = find_object(undistorted, settings)

        cv2.imshow('binary', undistorted)
        cv2.imshow('rgb', frame)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()