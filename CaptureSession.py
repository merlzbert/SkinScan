from abc import abstractmethod


class CaptureSession:
    def __init__(self, camera, projection, calibration, image_processing):
        # Initialize your imaging set-up
        self.camera = camera
        self.projection = projection
        self.calibration = None
        self.image_processing = image_processing

    @abstractmethod
    def capture(self):
        # To override: Capture procedure
        pass

    @abstractmethod
    def compute(self, calibration):
        # To override: Compute procedure
        pass
