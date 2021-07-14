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
            # Sample exposures that have been used in low-light settings with the Basler ace 2 camera:
            self.exposures = [30, 60, 100, 200, 400, 600, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 14000, 19000,
                              24000, 31000, 39000, 49000, 60000]
        else:
            self.exposures = exposures

    def capture(self):
        # For radiometric calibration a series of differently exposed images of the same object is required
        for exp in self.exposures:
            self.camera.setExposure(exp)
            self.camera.getImage(name='Radiometric/'+str(exp), calibration=True)

    def calibrate_HDR(self, smoothness=1000):
        # Call radiometric calibration
        # This computes and returns the camera response function and calculates a HDR image saved in path as PNG and .np
        self.g, le = self.radio_calib.get_camera_response(smoothness)
        self.radio_calib.plotCurve("Camera response")
        return self.g, le

    def load_calibration(self):
        self.radio_calib.load_calibration_data()
        self.radio_calib.plotCurve("Camera response")

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Calibrate radiometric calibrated images from a single exposure
        # Returns set of undistorted images and corresponding g function
        imgs, g = self.radio_calib.calibrate_image(exposure, path)
        return imgs, g


