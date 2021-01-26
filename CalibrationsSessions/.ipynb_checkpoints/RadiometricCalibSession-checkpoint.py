from abc import ABC
from CalibrationSession import CalibrationSession
from Calibrations import RadiometricCalibration


class RadiometricCalibSession(CalibrationSession, ABC):
    def __init__(self, camera, radio_calib, path='CalibrationImages/Radiometric', exposures=0):
        # Set camera, destination path
        self.camera = camera
        self.path = path
        self.g = None
        self.radio_calib = radio_calib
        if exposures == 0:
            self.exposures = [500, 750, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
                              22000]
        else:
            self.exposures = exposures

    def capture(self):
        # For radiometric calibration a series of differently exposed images of the same object is required
        for exp in self.exposures:
            self.camera.setExposure(exp)
            self.camera.getImage(name='Radiometric/'+str(exp), calibration=True)

    def calibrate_HDR(self, smoothness=10):
        # Call radiometric calibration
        # This computes and returns the camera response function and calculates a HDR image saved in path as PNG and .np
        self.g, g_n = self.radio_calib.CRC(smoothness)
        return self.g, g_n

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Calibrate radiometric calibrated images from a single exposure
        # Returns set of undistorted images and corresponding g function
        imgs, g = self.radio_calib.calibrate_image(exposure, path)
        return imgs, g


# self.exposures = [19, 54, 100, 298, 497, 803, 1599, 3198, 6402, 12798, 16003, 18007, 24005, 39993, 79986,
#                  159971, 300114]
# self.exposures = [19, 38, 76, 152, 304, 608, 1216, 2432, 4864, 9728, 19456, 38912, 77824, 155648, 311296,
#    622592, 1245184, 2490368, 4980736]

