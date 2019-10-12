#
# Helpers to automatically create OpenCV trackbars to change the settings
#

import cv2

def __none(x):
    pass

def foo(x):
    print(x)
    return x

def create(window_name, settings):
    """
    Create an OpenCV trackbar for each field in Settings.
    Fild structure: { i, value, max }
    Args:
        window_name Name of the OpenCV window to create trackbars
        settings Settings
    """
    cv2.namedWindow(window_name)
    props = sorted(settings.__dict__.items(), key=lambda x: x[1].i)
    for prop in props:
        cv2.createTrackbar(prop[0], window_name, prop[1].value, prop[1].max, __none)

def read(window_name, settings):
    """
    Update settings values from OpenCV trackbars positions.
    Refer "create" for more details
    Args:
        window_name Name of the OpenCV window to create trackbars
        settings Settings
    """
    for prop in settings.__dict__:
        settings.__dict__[prop].value = cv2.getTrackbarPos(prop, window_name)
