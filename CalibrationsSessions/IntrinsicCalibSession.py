from abc import ABC
from CalibrationSession import CalibrationSession


class IntrinsicCalibSession(CalibrationSession, ABC):
    def __init__(self, camera, intrinsic_calib, path='CalibrationImages', no=20):
        # Set camera
        self.camera = camera
        self.path = path
        self.no = no
        self.intrinsic_calib = intrinsic_calib

    def capture(self):
        # For an intrinsic calibration you are required to image the same Aruco markers from different 20 viewing
        # positions changing in rotation and translation of the camera
        for i in range(self.no):
            # Display camera stream
            self.camera.viewCameraStream()
            # Save images for calibration in .RAW and .PNG format
            filename = 'intr_' + str(i)
            self.camera.getImage(name='/Intrinsic/'+filename, calibration=True)
            print("Alternate the viewing direction of the camera...")

    def calibrate(self, imgFolder='CalibrationImages/Intrinsic', imgPattern="*.PNG"):
        # Call calibration in calibration object
        self.intrinsic_calib.calibration(imgFolder, imgPattern)

    def undistort_images(self, imgFolder='CapturedImages', imgPattern="*.PNG"):
        # Call undistort in calibration object
        self.intrinsic_calib.undistort(imgFolder, imgPattern)

    def undistort_npy(self, imgFolder='CapturedNumpyData', imgPattern="*.npy"):
        # Call undistort in calibration object
        self.intrinsic_calib.undistort(imgFolder, imgPattern)


