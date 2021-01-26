import numpy as np
import cv2
from Camera import Camera
import time
import matplotlib.pyplot as plt
import sys
from abc import ABC
# Just for Jupyter
from io import BytesIO, StringIO
from IPython.display import clear_output, Image, display, update_display
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
import PIL


class Internal(Camera, ABC):

    def __init__(self, exposure=0.1, white_balance=0, auto_focus=True, grayscale=True, internal_camera_port=1):
        # Setting and initializing the external camera
        self.port = internal_camera_port
        self.cap = cv2.VideoCapture(self.port)
        if self.cap.isOpened():
            # Wait for camera to initialize, otherwise black frames may occur
            time.sleep(0.5)
        if self.cap is None or not self.cap.isOpened():
            print("Warning: No internal webcam found.")
        # Get framerate and resolution of camera
        fps = self.getFPS()
        resolution = self.getResolution()
        self.grayscale = grayscale
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.hdr_exposures = None

    def getFPS(self):
        # Returns the frame rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps

    def setFPS(self, fps):
        # Sets the on frame rate
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.fps = fps

    def setAutoGain(self):
        raise NotImplementedError

    def getGain(self):
        raise NotImplementedError

    def setGain(self):
        raise NotImplementedError

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height

    def setResolution(self, resolution):
        # Sets a new resolution
        # May only support certain resolutions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.resolution = resolution

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        # Grab current frame
        ret, frame = self.cap.read()
        # Save images
        if saveImage:
            if calibration:
                cv2.imwrite('CalibrationImages/' + name + '.PNG', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite('CapturedImages/' + name + '.PNG', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if saveNumpy:
            if calibration:
                np.save('CalibrationNumpyData/' + name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                np.save('CapturedNumpyData/' + name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame

    def setAutoExposure(self):
        # Turn on auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def setExposure(self, exposure):
        # Not settable on most built in devices
        self.cap.set(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE, exposure)
        self.exposure = exposure

    def getExposure(self):
        # Not gettable on most built in devices
        return self.cap.get(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE)

    def viewCameraStream(self):
        # Live view
        while True:
            img = self.getImage(saveImage=False, saveNumpy=False)
            cv2.imshow('Internal Webcam Stream', img)
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                f = BytesIO()
                PIL.Image.fromarray(img).save(f, 'jpeg')
                obj = Image(data=f.getvalue())
                update_display(obj, display_id=dis.display_id)
                clear_output(wait=True)
        except KeyboardInterrupt:
            self.quit_and_open()

    def viewCameraStreamJupyterBokeh(self):
        try:
            # Live view in a Jupyter Notebook
            # Flickering occurs
            output_notebook()
            frame = self.getImage(saveImage=False, saveNumpy=False)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # because Bokeh expects a RGBA image
            frame = cv2.flip(frame, -1)  # because Bokeh flips vertically
            width = frame.shape[1]
            height = frame.shape[0]
            p = figure(x_range=(0, width), y_range=(0, height), width=width, height=height)
            myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
            show(p, notebook_handle=True)
            while True:
                frame = self.getImage(saveImage=False, saveNumpy=False)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.flip(frame, -1)
                myImage.data_source.data['image'] = [frame]
                push_notebook()
                time.sleep(2)
        except KeyboardInterrupt:
            self.quit_and_open()

    def quit_and_close(self):
        # Close camera
        self.cap.release()

    def quit_and_open(self):
        # Close camera
        self.cap.release()
        # Create new capture
        self.cap = cv2.VideoCapture(self.port)

    def getStatus(self):
        if self.cap is None or not self.cap.isOpened():
            print('Warning: unable to open video source port: ', self.port)


class External(Camera):

    def __init__(self, exposure=0.01, white_balance=0, auto_focus=False, grayscale=True, external_camera_port=0):
        # Setting and initializing the external camera
        self.port = external_camera_port
        self.cap = cv2.VideoCapture(self.port)
        if self.cap.isOpened():
            # Wait for camera to initialize, otherwise black frames may occur
            time.sleep(0.5)
        if self.cap is None or not self.cap.isOpened():
            print('Warning: unable to open external video source port: ', self.port)
        # Get framerate and resolution of camera
        fps = self.getFPS()
        resolution = self.getResolution()
        self.grayscale = grayscale
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.hdr_exposures = None

    def setAutoExposure(self):
        # Turn on auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def getFPS(self):
        # Returns the frame rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps

    def setFPS(self, fps):
        # Not settable on Mac
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.fps = fps

    def setAutoGain(self):
        raise NotImplementedError

    def getGain(self):
        raise NotImplementedError

    def setGain(self):
        raise NotImplementedError

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height

    def setResolution(self, resolution):
        # Sets new tuple of resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.resolution = resolution

    def setExposure(self, exposure):
        # Turn off auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Set exposure
        self.cap.set(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE, exposure)
        self.exposure = exposure

    def getExposure(self):
        # Return exposure value
        return self.cap.get(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE)

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        # Grab current frame
        ret, frame = self.cap.read()
        # Save images
        if saveImage:
            if calibration:
                cv2.imwrite('CalibrationImages/' + name + '.PNG', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite('CapturedImages/' + name + '.PNG', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if saveNumpy:
            if calibration:
                np.save('CalibrationNumpyData/' + name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                np.save('CapturedNumpyData/' + name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame

    def viewCameraStream(self):
        while True:
            img = self.getImage(saveImage=False, saveNumpy=False)
            cv2.imshow('Internal Webcam Stream', img)
            c = cv2.waitKey(1)
            if c != -1:
                # When everything done, release the capture
                self.quit_and_open()
                cv2.destroyAllWindows()
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
        self.cap.release()

    def quit_and_open(self):
        # Close camera
        self.cap.release()
        # Create new capture
        self.cap = cv2.VideoCapture(self.port)

    def getStatus(self):
        if self.cap is None or not self.cap.isOpened():
            print('Warning: unable to open external video source port: ', self.port)
