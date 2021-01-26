from abc import ABC
from CaptureSession import CaptureSession
from Projections import Pattern


class GradientShiftingCapture(CaptureSession, ABC):
    def __init__(self, camera, projection, image_processing, calibration=None, n=4):
        self.camera = camera
        self.projection = projection
        # Set gradient pattern as it's dimensions depends on the projection resolution
        self.projection.setPattern(Pattern.GradientPattern(projection.resolution))
        self.calibration = calibration
        self.image_processing = image_processing
        # Number of gradient images: default 4 - X&Y x R
        self.n = n

    def capture(self, red=1.0, green=1.0, blue=1.0):
        self.projection.pattern.createGradientXY(self.n, red=red, blue=blue, green=green)
        # Display patterns, take photos and save as np and jpg
        self.projection.displayPatterns(self.camera)

    def compute(self):
        if self.projection.root is not None:
            self.projection.quit_and_close()
        self.image_processing.loadData()
        gamma = self.calibration.radio_calib.gamma
        if self.camera.hdr_exposures is None:
            self.image_processing.computeNormalMapSingle(gamma)
        else:
            self.image_processing.computeNormalMapRadiance(gamma)
        self.image_processing.computeAlbedo()
        #self.image_processing.saveTiff()

    def calibrate(self, calibration):
        # Calibrate camera using a calibration object obtained from a Calibration Session
        self.calibration = calibration
        self.camera.setCalibration(calibration)