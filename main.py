from Cameras import Webcam, MachineVision, IpCam, NoCamera
from Projections import MainScreen
from CaptureSessions import DeflectometryCapture, GradientShiftingCapture, StepUpCapture
from Calibrations import RadiometricCalibration, IntrinsicCalibration
from CalibrationsSessions import RadiometricCalibSession, IntrinsicCalibSession
from Reconstructions import DeflectometryReconstruction, GradientShiftingReconstruction
from Visualization import Visualization
import numpy as np
import cv2
import Calibration
import matplotlib.pyplot as plt

#cam = IpCam.IpCam()


# Actual capture and compute code 

nph = 2

# Set-up your camera
# cam = Raspberry.RaspberryCam()
#cam = NoCamera.NoCamera()
#cam.setResolution((1200, 1920))
cam = MachineVision.Basler()
# Set-up your projector
projection = MainScreen.Screen()
# Camera settings

#cam.setExposure(200000.0)
#cam.setExposure(350000.0)
#cam.viewCameraStream()

# Radiometric calib
r_cal = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
cal = RadiometricCalibSession.RadiometricCalibSession(cam, r_cal)
#cal.calibrate_HDR()
#calib = Calibration.Calibration(radio_calib=r_cal)#, intr_calib=intr_calib)
#cam.setCalibration(calib)
#r_cal.load_calibration_data()
#r_cal.plotCurve('Grayscale')
"""
cam.getImage(name='single_calib')
hdr_exp = np.array([300000.0, 350000.0, 400000.0, 450000.0])
cam.setHDRExposureValues(hdr_exp)
cam.getHDRImage(name='hdr_calib')
"""
"""
cap = StepUpCapture.StepUpCapture(cam, projection)
# Capture images
cap.capture()
cap.compute()
"""
# cam.setAutoGain()



#hdr_exp = np.array([1000.0, 5000.0, 20000.0, 120000.0, 240000.0, 480000.0, 600000.0, 1000000.0])
#hdr_exp = np.array([480000.0, 600000.0, 800000.0])
#hdr_exp = np.array([200000.0, 210000.0, 240000.0])
#hdr_exp = np.array([60000.0, 100000.0, 600000.0, 2000000.0])
#hdr_exp = np.array([60000.0, 100000.0, 200000.0, 400000.0, 600000.0])
#hdr_exp = np.array([200000.0, 300000.0, 350000.0, 400000.0, 450000.0, 600000.0])
#cam.setHDRExposureValues(hdr_exp)
#cam.setExposure(240000.0)
# Intrinsic calib
intr_calib = IntrinsicCalibration.IntrinsicCalibration()
calibo = IntrinsicCalibSession.IntrinsicCalibSession(cam, intr_calib)
#calibo.capture()
# Set calibration object
calib = Calibration.Calibration(radio_calib=r_cal, intr_calib=intr_calib)

#projection.displayCalibrationPattern(cam)
# Set up image processing



#image_processing = DeflectometryReconstruction.DeflectometryReconstruction(nph=nph)
image_processing = GradientShiftingReconstruction.GradientShiftingReconstruction(n=nph)
# Set up your Deflectometry Session
#cap = DeflectometryCapture.DeflectometryCapture(cam, projection, image_processing, nph=nph)
cap = GradientShiftingCapture.GradientShiftingCapture(cam, projection, image_processing, n=nph)
cap.calibrate(calib)
# Capture images
#cap.capture(red=1.0, blue=1.0, green=1.0)
print("Capturing...")
#cam.quit_and_close()

# Compute results
cap.compute()
print("Computing...")
#image_processing.highPassFilter()
#image_processing.computePointCloud(((700, 300), (800, 300)))
#image_processing.computePointCloud(((400, 300), (1000, 300)))
# 16 skin models#
# Skin replic 0
#image_processing.computePointCloud(((831, 250), (717, 250)))
# Skin replic 1
#image_processing.computePointCloud(((833, 250), (689, 250)))
# Skin replic 2
#image_processing.computePointCloud(((817, 250), (677, 250)))
# Skin replic 3
#image_processing.computePointCloud(((835, 250), (687, 250)))
# Skin replic 4
#image_processing.computePointCloud(((831, 250), (703, 250)))
# Skin replic 5
#image_processing.computePointCloud(((825, 250), (701, 250)))

# Big Skin mole:
#image_processing.computePointCloud(((1000, 400), (350, 350)))
image_processing.computePointCloud(((1000, 400), (350, 350)))
#image_processing.computePointCloud(((1230-125, 250), (465-125, 250)))
# small Skin mole:
#image_processing.computePointCloud(((600, 250), (650, 250)))
#image_processing.computePointCloud(((755-125, 250), (775-125, 250)))

# Visualize results
vis = Visualization(cap.image_processing)
#vis.showPhaseMaps()
#vis.showAlbedo()
#vis.showAllImages()
#vis.showNormals()
#vis.showQuiverNormals(stride=30)
#image_processing.saveTiff()

#calibo.undistort_images('Results')