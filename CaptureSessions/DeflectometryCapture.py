from abc import ABC
from CaptureSession import CaptureSession
from Projections import Pattern


class DeflectometryCapture(CaptureSession, ABC):
    def __init__(self, camera, projection, image_processing, calibration=None, nph=4):
        self.camera = camera
        self.projection = projection
        # Set sinus pattern as it's frequency depends on the projection resolution
        self.projection.setPattern(Pattern.SinusPattern(projection.resolution))
        self.calibration = calibration
        self.image_processing = image_processing
        # Number of phase shifts
        self.nph = nph

    def capture(self, red=1.0, green=1.0, blue=1.0):
        self.projection.pattern.createSinusXY(self.nph, red=red, blue=blue, green=green)
        # Display patterns, take photos and save as np and jpg
        self.projection.displayPatterns(self.camera)
        self.camera.quit_and_close()

    def compute(self):
        self.image_processing.loadData()
        if self.nph > 4:
            self.image_processing.sinusoidalFitting()
        else:
            self.image_processing.computePhaseMaps()
        self.image_processing.computeNormalMap()
        #self.image_processing.saveTiff()

    def calibrate(self, calibration):
        # Calibrate camera using a calibration object obtained from a Calibration Session
        self.calibration = calibration
