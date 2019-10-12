import cv2

def __none(x):
    pass

def create(window_name, settings):
    """
    Create an OpenCV trackbar for each field in Settings.
    Fild structure: { name, value, max_value }
    Args:
        window_name Name of the OpenCV window to create trackbars
        settings Settings
    """
    cv2.namedWindow(window_name)
    for prop in settings.__dict__:
        cv2.createTrakbar(prop.name, window_name, prop.value, prop.max_value, __none)

def read(window_name, settings):
    """
    Update settings values from OpenCV trackbars positions.
    Refer "create" for more details
    Args:
        window_name Name of the OpenCV window to create trackbars
        settings Settings
    """
    for prop in settings.__dict__:
        settings.__dict__[prop].value = cv2.getTrackbarPos(settings.__dict__[prop].name, window_name)
