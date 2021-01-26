from abc import abstractmethod


class Projection:
    def __init__(self, resolution, frequency):
        # Tuple Resolution of display/screen in pixels (Width, Height)
        if resolution[0] < resolution[1]:
            self.resolution = reversed(resolution)
        else:
            self.resolution = resolution
        # Frequency Projections
        self.frequency = frequency
        self.calibration = None

    @abstractmethod
    def displayPatterns(self, patterns, camera):
        # To override
        pass

    @abstractmethod
    def getResolution(self):
        # To override
        pass

    @abstractmethod
    def setResolution(self, resolution):
        # To override
        pass

    @abstractmethod
    def quit_and_close(self):
        # To override
        pass

    @abstractmethod
    def getStatus(self):
        # To override
        pass
