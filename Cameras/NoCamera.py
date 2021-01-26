from Camera import Camera
from abc import ABC
import glob
import numpy as np
import cv2
from skimage import io
#from PySpinCapture import PySpinCapture as psc


class NoCamera(Camera, ABC):

    def __init__(self, exposure=0.01, white_balance=0, auto_focus=False, grayscale=True):
        #  TODO: pylon.FeaturePersistence.Save("test.txt", camera.GetNodeMap())
        fps = 0
        resolution = self.getResolution()
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.hdr_exposures = None

    def getAutoExposure(self):
        raise NotImplementedError

    def setAutoExposure(self):
        raise NotImplementedError

    def getFPS(self):
        raise NotImplementedError

    def setFPS(self, fps):
        raise NotImplementedError

    def setAutoGain(self):
        raise NotImplementedError

    def getGain(self):
        raise NotImplementedError

    def setGain(self, gain):
        raise NotImplementedError

    def getResolution(self):
        return 1200, 1920

    def setResolution(self, resolution):
        self.resolution = resolution

    def setSingleFrameCapture(self):
        raise NotImplementedError

    def setHDRExposureValues(self, exposures):
        raise NotImplementedError

    def setExposure(self, exposure):
        raise NotImplementedError

    def getExposure(self):
        raise NotImplementedError

    def getHDRImage(self, name='test', saveImage=True, saveNumpy=True, timeout=5000):
        raise NotImplementedError

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        raise NotImplementedError

    def viewCameraStream(self):
        raise NotImplementedError

    def viewCameraStreamSnapshots(self):
        raise NotImplementedError

    def quit_and_close(self):
        return 0

    def quit_and_open(self):
        return 0

    def getStatus(self):
        raise NotImplementedError