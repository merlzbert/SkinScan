from Cameras import Webcam, MachineVision
from Projections import MainScreen
from CaptureSessions import DeflectometryCapture, GradientShiftingCapture, StepUpCapture
from Calibrations import RadiometricCalibration, IntrinsicCalibration
from CalibrationsSessions import RadiometricCalibSession, IntrinsicCalibSession
from Reconstructions import DeflectometryReconstruction, GradientShiftingReconstruction
from Visualization import Visualization
import Calibration
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
# Set-up your camera
cam = MachineVision.Basler()# Radiometric calib
r_cal = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
cal = RadiometricCalibSession.RadiometricCalibSession(cam, r_cal)
r_cal.load_calibration_data()
hdr_exp = np.array([30000.0, 60000.0, 120000.0, 240000.0])
cam.setHDRExposureValues(hdr_exp)
# Set calibration object
calib = Calibration.Calibration(radio_calib=r_cal)
cam.setCalibration(calib)
"""
cam = MachineVision.Basler()
cam.setExposure(100000.0)
#cam.setAutoGain()
#cam.viewCameraStream()

# Set-up your projector
projection = MainScreen.Screen()

# Set up your Deflectometry Session
cap = StepUpCapture.StepUpCapture(cam, projection)
# Capture images
#cap.capture()
cap.compute()
print("Capturing...")

"""
path='CapturedNumpyData/capture_%i.npy'
frame_0 = np.load(path % 0)
frames = np.zeros((frame_0.shape[0], frame_0.shape[1], 50))
for i in range(0, 50):
    print(i)
    frames[..., i] = np.load(path % i)
print(frames.shape)
g_0 = frames[460:700, 610:700, :]
g_1 = frames[480:720, 730:800, :]
g_2 = frames[500:740, 830:920, :]
g_3 = frames[520:760, 960:1050, :]
for i in range(0, 50):
    g_0[..., i] = frames[460:700, 610:700, i]#/frames[460:700, 610:700, -1]
    g_1[..., i] = frames[480:720, 730:800, i]#/frames[480:720, 730:800, -1]
    g_2[..., i] = frames[500:740, 830:920, i]#/frames[500:740, 830:920, -1]
    g_3[..., i] = frames[520:760, 960:1050, i]#/frames[520:760, 960:1050, -1]

print(g_0.shape)

plt.imshow(g_0[..., 49], cmap='gray')
plt.show()
plt.imshow(g_1[..., 49], cmap='gray')
plt.show()
plt.imshow(g_2[..., 49], cmap='gray')
plt.show()
plt.imshow(g_3[..., 49], cmap='gray')
plt.show()

m_0 = np.mean(g_0, axis=(0, 1))
m_1 = np.mean(g_1, axis=(0, 1))
m_2 = np.mean(g_2, axis=(0, 1))
m_3 = np.mean(g_3, axis=(0, 1))

x = np.linspace(0, 255, 50)
plt.plot(x, m_0, label='White')
plt.plot(x, m_1, label='Flesh')
plt.plot(x, m_2, label='Brown')
plt.plot(x, m_3, label='Black')
plt.xlabel('Projected intensity')
plt.ylabel('Captured mean intensity')
plt.legend()
plt.show()


#cam.setAutoGain()
#cam.viewCameraStream()

"""

"""
# Set-up your projector
#projection = MainScreen.Screen()
resolution = (1920, 1080) #projection.getResolution()
# Create gradient

x = np.linspace(0, 1, resolution[0])
xD = np.linspace(0, resolution[0], resolution[0])
print(x.shape)
print(xD.shape)
y = np.linspace(0, 1, resolution[1])

coeff = [-6.33634985, 10.56707136, -4.06277237,  0.79012877, -0.01104522]
coeff1 = [-2.39958185,  7.09471598, -6.94586026,  3.19411896,  0.02044966]
x1 = coeff[0]*x**4 + coeff[1]*x**3 + coeff[2]*x**2 + coeff[3]*x + coeff[4]
x2 = coeff1[0]*x**4 + coeff1[1]*x**3 + coeff1[2]*x**2 + coeff1[3]*x + coeff1[4]
y1 = coeff[0]*y**3 + coeff[1]*y**2 + coeff[2]*y + coeff[3]
plt.plot(xD, x, label='Linear gradient')
plt.plot(xD, x1, label='Captured gradient')
plt.plot(xD, x2, label='Corrected gradient')
plt.legend()
plt.xlabel('Pixel on display along x-axis')
plt.ylabel('Normalized pixel intensity value')
plt.show()

# Create mesh grid
[gradientX, gradientY] = np.meshgrid(x1, y1)
cv2.imwrite('CalibrationImages/gradientXL.PNG', gradientX*255)
cv2.imwrite('CalibrationImages/gradientYL.PNG', gradientY*255)

cam = MachineVision.Basler()
exp = [1000, 2500, 5000, 7500, 10000, 15000, 20000]
for e in exp:
    cam.setExposure(e)
    cam.viewCameraStream()
    cam.getImage(name=str(e))

"""
"""
# Radiometric Calibration

cam = MachineVision.Basler()
#cam.setExposure(40000)
#exp = 32000
#cam.setExposure(exp)
#cam.viewCameraStream()
# cam = Webcam.Internal()

Exposure = []
ExposureImages = []
files = []
path = '/Users/Merlin/Desktop/CalibrationImages/ExposuresForHDR/capture_4'
for file in os.listdir(path):
    if file.endswith(".npy"):
        files.append(file)
files.sort(key=lambda x: int(x[:-4]))
files.sort()

# We used exposure time as filenames
for filename in files:
    image = np.load(path + '/' + filename)
    filename = os.path.splitext(filename)[0] + '\n'
    Exposure.append(int(filename))
    ExposureImages.append(image)

exps = np.array(Exposure)
imgs = np.array(ExposureImages)
print(exps)
print(imgs.shape)

r_cal = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
cal = RadiometricCalibSession.RadiometricCalibSession(cam, r_cal)
#cal.capture()
exposures = [30, 60, 100, 200, 400, 600, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 14000, 19000,
                              24000, 31000, 39000, 49000, 60000]
#g, g_n = cal.calibrate_HDR(smoothness=1000)
#r_cal.get_camera_response(1000, exposures)
r_cal.load_calibration_data()
#r_cal.CRC(1000)
#cal.load_calibration()
r_cal.get_HDR_image()
#z, zs = r_cal.CRC(1000)
#r_cal.get_camera_response()
#r_cal.get_HDR_image()
"""
"""
#cam.getImage()
img, g_exp = cal.calibrate_image(exp)
# use the created array to output your multiple images. In this case I have stacked 4 images vertically
f, axarr = plt.subplots(2,2)
i0 = axarr[0,0].imshow(img[0])
axarr[0,0].set_title('Normalized radiance map 0')
divider = make_axes_locatable(axarr[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i0, cax=cax)
i1 = axarr[0,1].imshow(img[1])
axarr[0,1].set_title('Normalized radiance map 1')
divider = make_axes_locatable(axarr[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i1, cax=cax)
i2 = axarr[1,0].imshow(img[2])
axarr[1,0].set_title('Normalized radiance map 2')
divider = make_axes_locatable(axarr[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i2, cax=cax)
i3 = axarr[1,1].imshow(img[3])
axarr[1,1].set_title('Normalized radiance map 3')
divider = make_axes_locatable(axarr[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i3, cax=cax)
plt.show()

"""
"""
np.save('CalibrationImages/Distorted/capture_0.npy', np.squeeze(img[0]))
np.save('CalibrationImages/Distorted/capture_1.npy', np.squeeze(img[1]))
np.save('CalibrationImages/Distorted/capture_2.npy', np.squeeze(img[2]))
np.save('CalibrationImages/Distorted/capture_3.npy', np.squeeze(img[3]))


cam.quit_and_close()

org = cv2.imread('/Users/Merlin/Desktop/Gradients/Screen/7500.PNG')
plt.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2, 1)
# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(org, cmap='gray')
axarr[1].imshow(img[0], cmap='gray')
plt.show()
"""

# Intrinsic calibration

cam = MachineVision.Basler()
# cam.setAutoExposure()
intr_calib = IntrinsicCalibration.IntrinsicCalibration()
calibo = IntrinsicCalibSession.IntrinsicCalibSession(cam, intr_calib)
# calibo.capture()
calibo.calibrate()

