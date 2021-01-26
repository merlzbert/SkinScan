from abc import ABC
from CalibrationSession import CalibrationSession


class GeometricCalibSession(CalibrationSession, ABC):
    def __init__(self, camera, projector, geometric_calib, path='CalibrationImages'):
        # Set camera
        self.camera = camera
        # Set projector
        self.projector = projector
        self.path = path
        self.geometric_calib = geometric_calib

    def capture(self):
        # For the extrinsic/geometric calibration you are required to print 8 Aruco markers
        # Checkerboard: CalibrationImages/6x6_40mm.pdf
        # Glue or mount them on the edges of you mirror as well as halfway between the edges
        # The Aruco markers will determine the position of the object relative to the camera
        # To determine the position of the screen, we display a checkerboard pattern on the screen
        # Make sure that the screen is visible on the mirror from the position of the camera
        self.projector.displayCalibrationPattern(self.camera)
        # The image is by default saved in CalibrationImages/Geometric

    def calibrate(self):
        # TODO: Include all calls necessary for calibration
        self.geometric_calib.calibrate()
        return self.camera
