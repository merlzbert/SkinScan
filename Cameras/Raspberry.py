import numpy as np
from picamera import PiCamera
from time import sleep
from abc import ABC
# if using an environment different to PyCharm
import sys
sys.path.insert(0, '..')
from Camera import Camera


class RaspberryCam(Camera, ABC):
    def __init__(self, exposure=0.1, white_balance=0, auto_focus=True, grayscale=False):
        # Setting and initializing the PiCam
        self.cap = PiCamera()
        # Wait for camera to initialize, otherwise black frames may occur
        sleep(0.5)
        if self.cap is None:
            print("Warning: No RaspberryCam found.")
        # Get framerate and resolution of camera
        fps = self.getFPS()
        resolution = self.getResolution()
        self.grayscale = grayscale
        self.hdr_exposures = None
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)

    def getFPS(self):
        # Returns the frame rate
        fps = self.cap.framerate
        return fps

    def setFPS(self, fps):
        # Sets framerate
        self.cap.framerate = fps
        self.fps = fps

    def setAutoGain(self):
        raise NotImplementedError

    def getGain(self):
        raise NotImplementedError

    def setGain(self):
        raise NotImplementedError

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        return self.cap.resolution

    def setResolution(self, resolution):
        # Sets new tuple of resolution
        self.cap.resolution = resolution
        self.resolution = resolution
        print("Camera resolution: ", self.resolution)

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False):
        # Grab current frame
        # TODO!!!!
        frame = np.empty((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        self.cap.capture(frame, 'rgb')
        # TODO: Save to jpg
        # if saveImage:
        #    self.cap.capture('CapturedImages/capture_%i.jpg' % count, 'jpeg')
        if saveNumpy:
            if calibration:
                np.save('CalibrationNumpyData/' + name, frame)
            else:
                np.save('CapturedNumpyData/' + name, frame)

    def setAutoExposure(self):
        # Turn on auto exposure
        self.cap.shutter_speed = 0

    def setExposure(self, exposure):
        # Set exposure value
        self.cap.shutter_speed = exposure
        self.cap.exposure_speed = exposure
        self.exposure = exposure

    def getExposure(self):
        # Returns exposure value
        return self.cap.exposure_speed

    def viewCameraStream(self, time=10):
        # Live preview
        self.cap.start_preview()
        input("Press Enter to continue")
        self.cap.stop_preview()

    def quit_and_close(self):
        # Close camera
        self.cap.close()

    def quit_and_open(self):
        # Close camera
        self.cap.close()
        # Create new capture
        self.cap = PiCamera()

    def getStatus(self):
        if self.cap is None:
            print('Warning: Unable to open Rasperry Cam: ')