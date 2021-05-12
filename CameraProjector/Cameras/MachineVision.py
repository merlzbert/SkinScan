from pypylon import pylon
from Camera import Camera
from abc import ABC
import numpy as np
import cv2
from skimage import io
from io import BytesIO
from IPython.display import clear_output, Image, display, update_display
import PIL
# from PySpinCapture import PySpinCapture as psc


class Basler(Camera, ABC):

    def __init__(self, exposure=0.01, white_balance=0, auto_focus=False, grayscale=True):
        #  TODO: pylon.FeaturePersistence.Save("test.txt", camera.GetNodeMap())
        # Setting and initializing the Basler camera
        self.cap = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cap.Open()
        if self.cap is None:
            print('Warning: unable to open external Basler camera')
        # Get framerate and resolution of camera
        fps = self.getFPS()
        resolution = self.getResolution()
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.hdr_exposures = None

    def getAutoExposure(self):
        # Returns if auto exposure is enabled
        return self.cap.ExposureAuto.GetValue()

    def setAutoExposure(self):
        # Turn on auto exposure
        self.cap.ExposureAuto.SetValue("Continuous")

    def getFPS(self):
        # Returns the frame rate
        return self.cap.AcquisitionFrameRate.GetValue()

    def setFPS(self, fps):
        # Sets frame rate
        self.cap.AcquisitionFrameRate.SetValue(fps)
        self.fps = fps

    def setAutoGain(self):
        # Set auto gain
        self.cap.GainAuto.SetValue("Once")

    def getGain(self):
        # Returns the set gain value
        return self.cap.Gain.GetValue()

    def setGain(self, gain):
        # Turn off auto gain
        self.cap.GainAuto.SetValue("Off")
        # Set gain value
        self.cap.Gain.SetValue(gain)

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        resolution = (self.cap.Width.GetValue(), self.cap.Height.GetValue())
        return resolution

    def setResolution(self, resolution):
        # Sets the image resolution
        self.cap.Width.SetValue(resolution[0])
        self.cap.Height.SetValue(resolution[1])
        self.resolution = resolution

    def setSingleFrameCapture(self):
        # Set single frame acquisition mode
        self.cap.AcquisitionMode.SetValue('SingleFrame')

    def setHDRExposureValues(self, exposures):
        self.hdr_exposures = exposures

    def setExposure(self, exposure):
        # Set auto exposure off
        self.cap.ExposureAuto.SetValue("Off")
        # Set exposure value in microseconds
        self.cap.ExposureTime.SetValue(exposure)
        self.exposure = exposure

    def getExposure(self):
        # Returns exposure value in microseconds
        return self.cap.ExposureTime.GetValue()

    def getHDRImage(self, name='test', saveImage=True, saveNumpy=True, timeout=5000):
        if self.calibration is None:
            print("Initialize calibration object of camera class first")
        self.cap.StartGrabbingMax(1)
        img = pylon.PylonImage()
        frames = []
        for e in self.hdr_exposures:
            self.setExposure(e)
            while self.cap.IsGrabbing():
                # Grabs photo from camera
                grabResult = self.cap.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    frame = grabResult.Array
                    img.AttachGrabResultBuffer(grabResult)
                grabResult.Release()
            frames.append(frame)
        hdr_frame = self.calibration.radio_calib.get_HDR_image(frames, self.hdr_exposures)
        if saveNumpy:
            np.save('CapturedNumpyData/' + name, hdr_frame)
        if saveImage:
            png_frame = (hdr_frame - np.min(hdr_frame)) / (np.max(hdr_frame) - np.min(hdr_frame))
            png_frame *= 255.0
            io.imsave('CapturedImages/' + name + '.PNG', png_frame.astype(np.uint8))
        return hdr_frame

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        try:
            # Take and return current camera frame
            self.cap.StartGrabbingMax(1)
            img = pylon.PylonImage()
            while self.cap.IsGrabbing():
                # Grabs photo from camera
                grabResult = self.cap.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    frame = grabResult.Array
                    img.AttachGrabResultBuffer(grabResult)
                grabResult.Release()
            # Save if desired
            if saveImage:
                if calibration:
                    filename = 'CalibrationImages/' + name + '.raw'
                    filenamePNG = 'CalibrationImages/' + name + '.PNG'
                    img.Save(pylon.ImageFileFormat_Raw, filename)
                    img.Save(pylon.ImageFileFormat_Png, filenamePNG)
                else:
                    filename = 'CapturedImages/' + name + '.PNG'
                    img.Save(pylon.ImageFileFormat_Png, filename)
            if saveNumpy:
                if calibration:
                    np.save('CalibrationNumpyData/' + name, frame)
                else:
                    np.save('CapturedNumpyData/' + name, frame)
            img.Release()
            self.cap.StopGrabbing()
            return frame
        except SystemError:
            self.quit_and_open()
            return None

    def viewCameraStream(self):
        # Display live view
        while True:
            cv2.namedWindow('Basler Machine Vision Stream', cv2.WINDOW_NORMAL)
            img = self.getImage(saveImage=False, saveNumpy=False)
            print("Max: ", np.max(img))
            print("Min: ", np.min(img))
            cv2.imshow('Basler Machine Vision Stream', img)
            c = cv2.waitKey(1)
            if c != -1:
                # When everything done, release the capture
                cv2.destroyAllWindows()
                break

    def viewCameraStreamSnapshots(self):
        # Display live view
        while True:
            cv2.namedWindow('Basler Machine Vision Stream', cv2.WINDOW_NORMAL)
            img = self.getImage(saveImage=False, saveNumpy=False)
            cv2.imshow('Basler Machine Vision Stream', img)
            c = cv2.waitKey(1)
            if c != -1:
                # When everything done, release the capture
                cv2.destroyAllWindows()
                self.quit_and_open()
                break

    def viewCameraStreamJupyter(self):
        # Live view in a Jupyter Notebook
        try:
            start = self.getImage(saveImage=False, saveNumpy=False)
            g = BytesIO()
            PIL.Image.fromarray(start).save(g, 'jpeg')
            obj = Image(data=g.getvalue())
            dis = display(obj, display_id=True)
            while True:
                img = self.getImage(saveImage=False, saveNumpy=False)
                if img is None:
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                f = BytesIO()
                PIL.Image.fromarray(img).save(f, 'jpeg')
                obj = Image(data=f.getvalue())
                update_display(obj, display_id=dis.display_id)
                clear_output(wait=True)
        except KeyboardInterrupt:
            self.quit_and_open()

    def quit_and_close(self):
        # Close camera
        self.cap.Close()

    def quit_and_open(self):
        # Close camera
        self.cap.Close()
        # Create new capture
        self.cap = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cap.Open()

    def getStatus(self):
        pylon.FeaturePersistence.Save("Basler_Specs.txt", self.cap.GetNodeMap())


