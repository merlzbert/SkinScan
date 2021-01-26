import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt


class RadiometricCalibration():

    def __init__(self, cam, path='CalibrationImages/Radiometric'):
        # # Set camera, destination path
        self.path = path
        self.cam = cam
        # Get image resolution:
        self.width, self.height = self.camera.getResolution()
        #self.width, self.height = 1920, 1200
        # Initialize g function with None for later
        self.g = None

    def CRC(self, l):
        # The whole program contains three parts: 1. Preprocess caputured images: 2. Solve g function; 3. Get HDR Image
        # We need to input path of caputured images, also the size of images for '.raw' file. Also we need to name these
        # files as their exposure time. l is the smooth parameter

        # This part is to preprocess captured images

        k = 0
        Exposure = []
        ExposureImages = []
        files = []
        for file in os.listdir(self.path):
            if file.endswith(".raw") or file.endswith(".Raw") or file.endswith(".RAW"):
                files.append(file)
        files.sort(key=lambda x: int(x[:-4]))
        files.sort()

        # We used exposure time as filenames
        for filename in files:
            image = np.fromfile(self.path + '/' + filename, dtype=np.uint8)
            # for .raw file, we need to know the picture shape in advance
            image.shape = self.width, self.height
            filename = os.path.splitext(filename)[0] + '\n'
            Exposure.append(int(filename))
            ExposureImages.append(image)
            # k is used to count the number of pictures
            k = k + 1

        ExposureNumber = k

        # Shrink the picture for faster calculation #
        size = (int(self.width * 0.01), int(self.height * 0.01))
        Z = np.zeros([size[0] * size[1], ExposureNumber], int)
        for i in range(ExposureNumber):
            x = cv2.resize(ExposureImages[i], size)

            Z[:, i] = cv2.resize(x, (size[0] * size[1], 1))

        B = np.log(Exposure) * 1.0
        w = np.ones([256, 1])
        for i in range(128):
            w[i] = i + 1
        for i in range(128, 255):
            w[i] = 256 - i

        # return Z,B,w,ExposureNumber,ExposureImages,Exposure. Those are what we need for following code ###

        # Next is to calculate g function #

        # Some explaination for solving g #
        #         gsolve.m - Solve for imaging system response function
        #
        #         Given a set of pixel values observed for several pixels in several
        #         images with different exposure times, this function returns the
        #         imaging system's response function g as well as the log film irradiance
        #         values for the observed pixels.
        #
        #         Assumes:
        #
        #         Zmin = 0
        #         Zmax = 255
        #
        #         Arguments:
        #
        #         Z(i, j) is the pixel values of pixel location number i in image j
        #         B(j)    is the log delta t, or log shutter speed, for image j
        #         l       is the lamda, the constant that determines the amout of smoothness
        #         w(z)    is the weighting function value for pixel value z
        #
        #         Returns:
        #
        #         g(z)    is the log exposure corresponding to pixel value z
        #         lE(i)   is the log film irradiance at pixel location i

        n = 256
        A = np.zeros([np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 0)])
        b = np.zeros([np.size(A, 0), 1])

        # Include the data-fitting equations
        k = 1
        for i in range(np.size(Z, 0)):
            for j in range(np.size(Z, 1)):
                wij = w[Z[i, j]]
                A[k, Z[i, j]] = wij
                A[k, n + i] = -wij
                b[k, 0] = wij * B[j]
                k = k + 1

        # Fix the curve by setting its middle value to 0
        A[k, 129] = 1
        k = k + 1

        # Include the smoothness equations
        for i in range(n - 2):
            A[k, i] = l * w[i + 1]
            A[k, i + 1] = -2 * l * w[i + 1]
            A[k, i + 2] = l * w[i + 1]
            k = k + 1

        # Solve the system using SVD
        A = np.linalg.pinv(A)
        x = A.dot(b)
        self.g = x[0:256]
        # return g. That's what we need #

        # Next is to get HDR image #

        size = (int(self.height * 1), int(self.width * 1))
        EE = np.zeros([size[0] * size[1], 1])
        sumw = np.zeros([size[0] * size[1], 1], int)

        for i in range(ExposureNumber):
            t = ExposureImages[i].flatten()

            EE = EE + w[t] * (self.g[t] - np.log(Exposure[i]))

            sumw = sumw + w[t]

        lE = np.reshape(EE / sumw, size)
        cv2.imwrite(self.path + '/hdr.PNG', lE)
        # Save the data and plot #
        np.save('CalibrationNumpyData/g.npy', self.g)
        np.save('CalibrationNumpyData/lE.npy', lE)
        plt.figure(1)
        plt.plot(np.linspace(0, 255, 256), self.g)
        plt.show()
        plt.figure(2)
        plt.imshow(lE, cmap='gray')
        plt.show()
        g_n = (self.g - min(self.g)) / (max(self.g) - min(self.g))
        return self.g, g_n

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Create list of calibrated images
        images = []
        # Exposure in microseconds -> convert to seconds
        exp = exposure * (10 ** -6)
        g = np.exp(self.g)
        # Load images
        imgFileList = self.readFileList(path)
        # Iterate over images to be calibrated
        for i in imgFileList:
            # Read image and convert to grayscale if necessary
            img = cv2.imread(i)
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # Applying the Debevec Algorithm
            # Eq. 5 Debevec et. al.
            calibrated_image = g[gray]
            calibrated_image = np.exp(calibrated_image - np.log(exp))
            cv2.imwrite(path + '/RadianceMaps' + i + '.PNG', calibrated_image)
            images.append(calibrated_image)
        return images, g

    @staticmethod
    def readFileList(imgFolder, ImgPattern="*.PNG"):
        imgFileList = glob.glob(os.path.join(imgFolder, ImgPattern))
        imgFileList.sort()
        return imgFileList
