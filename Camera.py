from abc import abstractmethod


class Camera:
    def __init__(self, exposure, white_balance, auto_focus, fps, resolution, grayscale):
        # Exposure passed as float in seconds
        self.exposure = exposure
        # White balanced passed as a float
        self.white_balance = white_balance
        # Auto_focus passed as boolean
        self.auto_focus = auto_focus
        # FPS in float
        self.fps = fps
        # Resolution as tuple (Width, Height)
        self.resolution = resolution
        # Grayscale in boolean
        self.grayscale = grayscale
        # Capture object may be in cv2.capture, pypylon, PySpin etc.
        self.capture = None
        # Calibration object
        self.calibration = None

    @abstractmethod
    def getImage(self):
        # To override: Capture image, return frame and save in corresponding folder in specified file format
        pass

    @abstractmethod
    def setExposure(self, exposure):
        # To override
        pass

    @abstractmethod
    def getExposure(self):
        # To override
        pass

    @abstractmethod
    def getFPS(self):
        # To override
        pass

    @abstractmethod
    def setFPS(self):
        # To override
        pass

    @abstractmethod
    def setAutoGain(self):
        # To override
        pass

    @abstractmethod
    def getGain(self):
        # To override
        pass

    @abstractmethod
    def setGain(self):
        # To override
        pass

    @abstractmethod
    def getResolution(self):
        # To override
        pass

    @abstractmethod
    def setResolution(self):
        # To override
        pass

    @abstractmethod
    def viewCameraStream(self):
        # To override
        pass

    @abstractmethod
    def quit_and_close(self):
        # To override
        pass

    @abstractmethod
    def quit_and_open(self):
        # To override
        pass

    @abstractmethod
    def getStatus(self):
        # To override
        pass

    @abstractmethod
    def setCalibration(self, calibration):
        # Set the calibration object
        self.calibration = calibration





