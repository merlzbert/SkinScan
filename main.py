from Cameras import Webcam, MachineVision
from Projections import MainScreen
from CaptureSessions import GradientIlluminationCapture
from Calibrations import RadiometricCalibration, IntrinsicCalibration
from CalibrationsSessions import RadiometricCalibSession, IntrinsicCalibSession
from Reconstructions import GradientIlluminationReconstruction
from Visualization import Visualization
import numpy as np
import cv2
import Calibration
import matplotlib.pyplot as plt

# Actual capture and compute code 

nph = 2

# Set-up your camera
cam = MachineVision.Basler()
# Set-up your projector
projection = MainScreen.Screen()
# Camera settings
#cam.viewCameraStream()

# Radiometric calib
r_cal = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
cal = RadiometricCalibSession.RadiometricCalibSession(cam, r_cal)
# cal.calibrate_HDR()
# cam.setCalibration(calib)
# r_cal.load_calibration_data()
# r_cal.plotCurve('Grayscale')


# Intrinsic calib
intr_calib = IntrinsicCalibration.IntrinsicCalibration()
calibo = IntrinsicCalibSession.IntrinsicCalibSession(cam, intr_calib)
#calibo.capture()
# Set calibration object
calib = Calibration.Calibration(radio_calib=r_cal, intr_calib=intr_calib)


# Set up image processing
image_processing = GradientIlluminationReconstruction.GradientIlluminationReconstruction(n=nph)
# Set up your Deflectometry Session
cap = GradientIlluminationCapture.GradientIlluminationCapture(cam, projection, image_processing, n=nph)
cap.calibrate(calib)
# Capture images
#cap.capture(red=1.0, blue=1.0, green=1.0)
print("Capturing...")
#cam.quit_and_close()

# Compute results
cap.compute()
print("Computing...")

# Visualize results
vis = Visualization(cap.image_processing)
vis.showPhaseMaps()
vis.showAlbedo()
vis.showAllImages()
vis.showNormals()
#vis.showQuiverNormals(stride=30)
#image_processing.saveTiff()

#calibo.undistort_images('Results')
