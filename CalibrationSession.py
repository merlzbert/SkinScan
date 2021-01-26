from abc import abstractmethod


class CalibrationSession:
    def __init__(self, camera, path='CalibrationImages'):
        # Initialize the camera to calibrate
        self.camera = camera
        # This path points to where images for calibration are stored
        self.path = path

    @abstractmethod
    def capture(self):
        # To override: Image capture procedure for different calibrations
        pass

    @abstractmethod
    def calibrate(self, calibration):
        # To override: Calibration computation
        pass


