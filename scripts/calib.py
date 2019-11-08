#
# Helpers to work with camera calibraiton data
#
import yaml
import numpy as np

class CameraCalibration:
    def __init__(self):
        pass

    @staticmethod
    def load(filename):
        self = CameraCalibration()
        try:
            with open(filename, 'r') as f:
                calib = yaml.load(f)
                width = calib['image_width']
                height = calib['image_height']
                self.size = (width, height)
                self.camera_matrix = CameraCalibration.__parse_array(calib['camera_matrix'])
                self.distortion = CameraCalibration.__parse_array(calib['distortion_coefficients'])
                return self
        except Exception as ex:
            print('Failed to load calibration file {}: {}'.format(filename, str(ex)))
            return None

    # Read matrix with given shape from file
    @staticmethod
    def __parse_array(node):
        return np.reshape(np.array(node['data']), (node['rows'], node['cols']))