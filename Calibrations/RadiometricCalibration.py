import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt


class RadiometricCalibration:

    def __init__(self, resolution, gamma=0.57, sampling_points=1000, path='CalibrationImages/Radiometric'):
        # # Set camera, destination path
        self.path = path
        # Get image resolution:
        self.width, self.height = resolution
        # Amount of sample points per image - to speed up calculation
        self.sampling_points = sampling_points
        # Initialize g function with None for later same for log exposure values
        # Gamma correction:
        self.gamma = gamma
        self.g = None
        self.w = None
        self.le = None
        # Raw captured data
        self.raw_data = None
        # Down-sampled data
        self.raw_samples = None
        # Exposure times
        self.exposures = None

    def load_calibration_data(self):
        # Check if calibration file already exists
        if os.path.exists('CalibrationNumpyData/radiometric.npz'):
            # Load g function, log exposure, weighting function, exposures, raw samples
            data = np.load('CalibrationNumpyData/radiometric.npz')
            self.g = data['g_function']
            self.le = data['log_exposures']
            self.w = data['w_function']
            self.exposures = data['exposures']
            self.raw_samples = data['samples']
        else:
            print("Capture and calibrate camera first")

    def load_raw_data(self):
        # Loading raw data files
        k = 0
        # Empty lists for images and exposure times
        Exposure = []
        self.raw_data = []
        files = []
        for file in os.listdir(self.path):
            # Only use .raw files
            if file.endswith(".raw") or file.endswith(".Raw") or file.endswith(".RAW"):
                files.append(file)
        # Sort files depending on their exposure time from lowest to highest
        files.sort(key=lambda x: int(x[:-4]))
        # We used exposure time as filenames
        for filename in files:
            image = np.fromfile(self.path + '/' + filename, dtype=np.uint8)
            # for .raw file, we need to know the picture shape in advance
            image.shape = self.width, self.height
            filename = os.path.splitext(filename)[0] + '\n'
            Exposure.append(int(filename))
            self.raw_data.append(image)
            # k is used to count the number of pictures
            k = k + 1

        ExposureNumber = k

        # Shrink the number of samples #
        Z = np.zeros([self.sampling_points, ExposureNumber], int)
        # Choose random sample points within image
        row = np.random.randint(self.width, size=self.sampling_points)
        col = np.random.randint(self.height, size=self.sampling_points)
        for i in range(ExposureNumber):
            Z[:, i] = self.raw_data[i][row, col]
        # Initialize to raw_samples
        self.raw_samples = Z
        exps = np.sort(np.array(Exposure))
        # Check if loaded exposure values match the predefined values
        self.exposures = exps
        print("Radiometric raw data loaded...")

    def plotCurve(self, title):
        """
        This function will plot the curve of the solved G function and the measured pixels. You don't need to return anything in this function.
        Input
        solveG: A (256,1) array. Solved G function generated in the previous section.
        LE: Log Erradiance of the image.
        logexpTime: (k,) array, k is the number of input images. Log exposure time.
        zValues: m*n array. m is the number of sampling points, and n is the number of input images. Z value generated in the previous section.

        Please note that in this function, we only take z value in ONLY ONE CHANNEL.

        title: A string. Title of the plot.
        """
        logexpTime = np.log(self.exposures*(10**-6))
        plt.title(title)
        plt.xlabel('Log exposure')
        plt.ylabel('Pixel intensity value')
        LEx = np.expand_dims(self.le, axis=1)
        LEx = np.repeat(LEx, logexpTime.shape[0], axis=1)
        logx = np.expand_dims(logexpTime, axis=1)
        logx = np.swapaxes(logx, 0, 1)
        logx = np.repeat(logx, self.le.shape[0], axis=0)
        x = logx + LEx
        plt.plot(x, self.raw_samples, 'ro', alpha=0.5)
        plt.plot(self.g, np.linspace(0, 255, 256))
        plt.show()

    def get_camera_response(self, smoothness):
        """
        Some explanation for solving g:
        Given a set of pixel values observed for several pixels in several
        images with different exposure times, this function returns the
        imaging system's response function g as well as the log film irradiance
        values for the observed pixels.
        Assumes:
        Zmin = 0
        Zmax = 255
        Arguments:
        self.raw_sample - Z(i, j)  is the pixel values of pixel location number i in image j
        self.exposure B(j)    is the log delta t, or log shutter speed, for image j
        l       is the lamda, the constant that determines the amount of smoothness
        w(z)    is the weighting function value for pixel value z
        Returns:
        g(z)    is the log exposure corresponding to pixel value z
        lE(i)   is the log film irradiance at pixel location i
        """
        # Load raw data
        if self.raw_data is None:
            self.load_raw_data()
        Z = self.raw_samples.astype(np.int)
        # Convert exposure to log exposure
        B = np.log(self.exposures*(10**-6))
        # Next is to calculate g function #
        n = 256
        # Create weighting function - hat like
        """
        self.w = np.ones([256, 1])
        for i in range(128):
            self.w[i] = i + 1
        for i in range(128, 255):
            self.w[i] = 256 - i
        """
        self.w = np.ones((n, 1)) / n
        m = Z.shape[0]
        p = Z.shape[1]
        A = np.zeros((m * p + n + 1, n + m))
        b = np.zeros((A.shape[0], 1))
        k = 0
        # Data fitting equations
        for i in range(m):
            for j in range(p):
                wij = self.w[Z[i, j]]
                A[k, Z[i, j]] = wij
                A[k, n + i] = -wij
                b[k, 0] = wij * B[j]
                k += 1
        # Fix the curve by setting its middle value to 0
        A[k, 128] = 1
        k = k + 1
        # Include smoothness equations
        for i in range(n - 2):
            A[k, i] = smoothness * self.w[i + 1]
            A[k, i + 1] = -2 * smoothness * self.w[i + 1]
            A[k, i + 2] = smoothness * self.w[i + 1]
            k = k + 1
        # Solve the system using SVD
        x = np.linalg.lstsq(A, b, rcond=None)
        x = x[0]
        self.g = x[0:n]
        lE = x[n:x.shape[0]]
        self.le = lE.squeeze()
        # Save g function, exposures, etc for loading
        np.savez('CalibrationNumpyData/radiometric.npz', g_function=self.g, log_exposures=self.le[::10],
                 w_function=self.w, exposures=self.exposures, samples=self.raw_samples[::10, :])
        return self.g, self.le

    def get_HDR_image(self, images=None, exposures=None):
        # If images is None, take radiometric calibration images
        if images is None:
            if self.raw_data is None:
                self.load_raw_data()
                images = self.raw_data
            else:
                images = self.raw_data
        # If g function is None, load calibration
        if self.g is None:
            self.load_calibration_data()
        # Override exposure values
        if exposures is not None:
            self.exposures = exposures
        # Compute log exposure image
        # Initialize flatten
        size = (int(self.height * 1), int(self.width * 1))
        EE = np.zeros([size[0] * size[1], 1])
        sumw = np.zeros([size[0] * size[1], 1], int)
        # Convert exposure from microseconds to seconds
        exp_sec = self.exposures * (10 ** -6)
        num_exp = self.exposures.shape[0]
        for i in range(num_exp):
            t = images[i].flatten()
            EE = EE + self.w[t] * (self.g[t] - np.log(exp_sec[i]))
            #EE = EE + (self.g[t] - np.log(exp_sec[i]))
            sumw = sumw + self.w[t]
        # Reshape
        lE = np.reshape(EE / sumw, size)
        #lE = np.reshape(EE / num_exp, size)
        # Take exponent to get exposure for each pixel
        exposure_image = np.exp(lE)
        return exposure_image

    def calibrate_image(self, exposure, path):
        # UNUSED
        # Create list of calibrated images
        images = []
        # Exposure in microseconds -> convert to seconds
        exp = exposure * (10 ** -6)
        g = np.exp(self.g)
        # Load images
        imgFileList = self.readFileList(path)
        # Idx
        k = 0
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
            #calibrated_image = np.exp(calibrated_image - np.log(exp))
            calibrated_image = calibrated_image - np.log(exp)
            #calibrated_image *= 255.0 / calibrated_image.max()
            images.append(calibrated_image)
            k += 1
        # Normalize by last captured image, which represents the object lit by a constant (255) illumination pattern
        illuminated_radiance = images.pop()
        for r in range(len(images)):
            n_img = images[r]/illuminated_radiance
            #n_img = images[r]
            # Gamma correction
            #n_img = (n_img - np.min(n_img)) / (np.max(n_img) - np.min(n_img))
            n_img = self.apply_gamma_curve(n_img, gamma=0.4)
            cv2.imwrite(path + '/RadianceMaps/capture' + str(r) + '.PNG', n_img*255)
            np.save(path + '/RadianceMaps/capture_' + str(r) + '.npy', n_img)
            images[r] = n_img
        return images, g

    @staticmethod
    def scaleBrightness(E):
        # Unused
        """
        Brightness scaling function, which will scale the values on the radiance map to between 0 and 1

        Args:
            E: An m*n*3 array. m*n is the size of your radiance map, and 3 represents R, G and B channel. It is your plotted Radiance map (don't forget to use np.exp function to get it back from logorithm of radiance!)
        Returns:
            ENomrMap: An m*n*3 array. Normalized radiance map, whose value should between 0 and 1
        """
        res = np.zeros(E.shape)
        for c in range(E.shape[2]):
            res[:, :, c] = (E[:, :, c] - np.min(E[:, :, c])) / (np.max(E[:, :, c]) - np.min(E[:, :, c]))
        return res

    @staticmethod
    def apply_gamma_curve(E, gamma=0.4):
        # Unused
        """
        apply gamma to the curve through raising E to the gamma.

        Args:
            E: An m*n*3 array. m*n is the size of your radiance map, and 3 represents R, G and B channel. It is your plotted Radiance map (don't forget to use np.exp function to get it back from logorithm of radiance!)
            gamma: a float value that is representative of the power to raise all E to.
        Returns:
            E_gamma: E modified by raising it to gamma.
        """
        return E ** gamma

    @staticmethod
    def readFileList(imgFolder, ImgPattern="*.PNG"):
        imgFileList = glob.glob(os.path.join(imgFolder, ImgPattern))
        imgFileList.sort()
        print(imgFileList)
        return imgFileList

