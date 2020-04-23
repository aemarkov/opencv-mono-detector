#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import yaml
import os

from settings import Settings
import ui

import rospy
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber

IMAGE_TOPIC = '/image'
CAMERA_INFO_TOPIC='/camera_info'
MARKER_TOPIC='/direction'
POS_TOPIC='/object'
FRAME='base_link'

def init_argparse():
    parser = argparse.ArgumentParser(description='Find object with given color and calculate ray or position')
    parser.add_argument('--capture', type=int, default=0, help='OpenCV video capture')
    parser.add_argument('--config', required=True, help='Path to the configuration file to open/save')
    parser.add_argument('--size', type=float, help='Object size (diameter) in meters')
    parser.add_argument('--gui', choices=['base', 'full'], help='Show GUI')
    parser.add_argument('--tuning', action='store_true', help='Run in tuning mode, no projection')
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
    if moments['m00'] < 1:
       return None
    center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

    if args.gui == 'base' or args.gui == 'full':
        cv2.drawContours(frame, contours, index, (0, 255, 0), 3)
        cv2.circle(frame, center, 5, (0, 255, 255), 3)

    return center

# Reproject 2D coords to the 3D
def reproject(center, camera_info):
    # camera matrix - 3x3 matrix in row-major order
    u, v = center
    u = camera_info.width - u
    Cx = camera_info.K[0*3 + 2]
    Cy = camera_info.K[1*3 + 2]
    fx = camera_info.K[0*3 + 0]
    fy = camera_info.K[1*3 + 1]
    return np.array([(u - Cx)/fx, (v - Cy)/fy, 1])

# Publish object position to ROS
def publish(vector, pose_pub, line_pub):
    x, y, z = vector
    point = Point(z, x, y)

    pose = PoseStamped()
    pose.header.frame_id = FRAME
    pose.pose.position = point

    marker = Marker()
    marker.type = Marker.LINE_LIST
    marker.header.frame_id = FRAME
    marker.scale = Vector3(0.01, 0, 0)
    marker.color = ColorRGBA(1, 0, 0, 1)
    marker.points.append(Point(0, 0, 0))
    marker.points.append(point)

    pose_pub.publish(pose)
    line_pub.publish(marker)

def image_callback(image, camera_info):
    if args.gui == 'full':
            ui.read('controls', config)

    frame = bridge.imgmsg_to_cv2(image, "bgr8")

    # Find objects and publish
    center = find_object(frame, config, args)
    if center != None and not args.tuning:
        point_3d = reproject(center, camera_info)
        publish(point_3d, pose_pub, line_pub)

    if args.gui == 'base' or args.gui == 'full':
        cv2.imshow('rgb', frame)

    key = cv2.waitKey(1)
    if key == 115:
        print('Saving...')
        config.store(args.config)
    if key == 27:
        print('Exit')
        exit(0)



if __name__ == "__main__":
    parser = init_argparse()
    args, unknown = parser.parse_known_args()

    rospy.init_node('object_finder')
    pose_pub = rospy.Publisher(POS_TOPIC, PoseStamped, queue_size=1)
    line_pub = rospy.Publisher(MARKER_TOPIC, Marker, queue_size=1)

    tss = TimeSynchronizer([
        Subscriber(IMAGE_TOPIC, Image),
        Subscriber(CAMERA_INFO_TOPIC, CameraInfo)], 10)
    tss.registerCallback(image_callback)

    bridge = CvBridge()

    print('----------------------------------------------')
    print('OpenCV object detector')
    print('Controls:')
    print('    - S     Save current config')
    print('    - ESC   Exit without saving current config')
    print('----------------------------------------------')

    config = Settings.load(args.config, default=make_default_config())
    if config == None:
        exit(1)

    if args.gui == 'full':
        ui.create('controls', config)

    rospy.spin()
