from abc import ABC
import numpy as np
from PIL import Image, ImageTk
from Projection import Projection


class GradientPattern(Projection, ABC):
    def __init__(self, resolution, frequency=0):
        super().__init__(resolution, frequency)
        self.patterns = None
        # Set frequency depending on screen resolution

    def createGradientXY(self, n=2, red=1.0, green=1.0, blue=1.0):
        #coeff = [-2.39958185, 7.09471598, -6.94586026, 3.19411896, 0.02044966]
        # Number gradient shifts n:
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], 3, n*2+1))
        # Create constant pattern
        c = np.ones((self.resolution[1], self.resolution[0]))
        # Create gradient
        x = np.linspace(0, 1, self.resolution[0])
        y = np.linspace(0, 1, self.resolution[1])
        #x = coeff[0] * x ** 4 + coeff[1] * x ** 3 + coeff[2] * x ** 2 + coeff[3] * x + coeff[4]
        #y = coeff[0] * y ** 4 + coeff[1] * y ** 3 + coeff[2] * y ** 2 + coeff[3] * y + coeff[4]
        # Reverse gradient
        xR = np.flipud(x)
        yR = np.flipud(y)
        # Create mesh grid
        [gradientX, gradientY] = np.meshgrid(x, y)
        [gradientXR, gradientYR] = np.meshgrid(xR, yR)

        # Set up pattern
        self.patterns[:, :, 0, 0] = red * gradientX
        self.patterns[:, :, 1, 0] = green * gradientX
        self.patterns[:, :, 2, 0] = blue * gradientX
        self.patterns[:, :, 0, 1] = red * gradientXR
        self.patterns[:, :, 1, 1] = green * gradientXR
        self.patterns[:, :, 2, 1] = blue * gradientXR
        self.patterns[:, :, 0, 2] = red * gradientY
        self.patterns[:, :, 1, 2] = green * gradientY
        self.patterns[:, :, 2, 2] = blue * gradientY
        self.patterns[:, :, 0, 3] = red * gradientYR
        self.patterns[:, :, 1, 3] = green * gradientYR
        self.patterns[:, :, 2, 3] = blue * gradientYR
        self.patterns[:, :, 0, 4] = red * c
        self.patterns[:, :, 1, 4] = green * c
        self.patterns[:, :, 2, 4] = blue * c
        return self.patterns


class SinusPattern(Projection, ABC):
    def __init__(self, resolution, frequency=0):
        super().__init__(resolution, frequency)
        self.x = np.linspace(1, self.resolution[0], self.resolution[0])
        self.y = np.linspace(1, self.resolution[1], self.resolution[1])
        [self.X, self.Y] = np.meshgrid(self.x, self.y)
        self.patterns = None
        # Set frequency depending on screen resolution
        self.frequency = 1 / self.resolution[0]

    def createSinusXY(self, nph, red=1.0, green=1.0, blue=1.0):
        # Number of phase shifts: nph
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], 3, nph * 2))
        # Set up phase map
        phaseX = self.frequency * self.X
        phaseY = self.frequency * self.Y
        # Loop of number_of_phase_shifts to create sinusoidal patterns in X and Y direction
        for i in range(nph):
            # Calculate Phase Shifts in X and Y direction
            # Simple formula to create fringes between 0 and 1:
            phase_shift = i * 2 * np.pi / nph
            self.patterns[:, :, 0, i] = red * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 1, i] = green * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 2, i] = blue * (np.cos(phaseX - phase_shift) + 1) / 2
            self.patterns[:, :, 0, i + nph] = red * (np.cos(phaseY - phase_shift) + 1) / 2
            self.patterns[:, :, 1, i + nph] = green * (np.cos(phaseY - phase_shift) + 1) / 2
            self.patterns[:, :, 2, i + nph] = blue * (np.cos(phaseY - phase_shift) + 1) / 2
        return self.patterns


class StepPattern(Projection, ABC):

    def __init__(self, resolution, frequency=0):
        super().__init__(resolution, frequency)
        self.patterns = None
        # Set frequency depending on screen resolution

    def createStep(self, n=50):
        # Number gradient shifts n:
        # Set up pattern list to store phase shift images (X and Y direction)
        self.patterns = np.zeros((self.resolution[1], self.resolution[0], n+1))
        # Create calibrated values
        x = np.linspace(0, 1, 50)
        #coeff1 = [-2.39958185,  7.09471598, -6.94586026,  3.19411896,  0.02044966]
        #y = coeff1[0]*x**4 + coeff1[1]*x**3 + coeff1[2]*x**2 + coeff1[3]*x + coeff1[4]
        # Create constant patterns
        for i in range(n):
            self.patterns[:, :, i] = np.ones((self.resolution[1], self.resolution[0])) * x[i]
        return self.patterns
