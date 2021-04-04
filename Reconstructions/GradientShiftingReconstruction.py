from abc import ABC
from Reconstructions import Mesh
from ImageProcessing import ImageProcessing
import wavepy
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
from tifffile import imsave
import wavepy

np.seterr(divide='ignore', invalid='ignore')


class GradientShiftingReconstruction(ImageProcessing, ABC):

    def __init__(self, capture_path='CapturedNumpyData/capture_%i.npy', n=2):
        super().__init__(capture_path)
        # Number of gradient images per viewing direction
        self.n = n
        self.frames_x = None
        self.frames_y = None
        self.frame_reflectivity = None
        self.diff_x = None
        self.diff_y = None
        self.magH = None
        self.magV = None
        self.normals = None
        self.depth = None
        self.albedo = None

    def loadData(self):
        #  Load first frame for dimensions and expand along 2nd axis to create a 3D array for all phase shifts
        self.frames_x = np.load(self.path % 0)
        self.frames_y = np.load(self.path % self.n)
        # RGB frames
        if len(self.frames_x.shape) == 3 or len(self.frames_y.shape) == 3:
            for i in range(1, self.n):
                self.frames_x = np.stack((self.frames_x, np.load(self.path % i)), axis=3)
                self.frames_y = np.stack((self.frames_y, np.load(self.path % (i + self.n))), axis=3)
            self.frame_reflectivity = np.load(self.path % (self.n * 2))
        # Grayscale frames
        else:
            self.frames_x = np.expand_dims(self.frames_x, axis=2)
            self.frames_y = np.expand_dims(self.frames_y, axis=2)
            for i in range(1, self.n):
                self.frames_x = np.dstack((self.frames_x, np.load(self.path % i)))
                self.frames_y = np.dstack((self.frames_y, np.load(self.path % (i + self.n))))
            self.frame_reflectivity = np.load(self.path % (self.n * 2))

    def saveTiff(self):
        # Saves normals and captured images in tiff file format
        cv2.imwrite('normals.tif',
                    cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        for i in range(self.n):
            imsave('captureX_%i.tif' % i, self.frames_x[:, :, i].astype(np.float16))
            imsave('captureY_%i.tif' % i, self.frames_y[:, :, i].astype(np.float16))

    def computeAlbedo(self):
        # Average over all frames to retrieve an albedo estimation
        # RGB frames
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            stack = np.concatenate((self.frames_x, self.frames_y), axis=3)
            self.albedo = np.mean(stack, axis=3)
        # Grayscale
        else:
            stack = np.dstack((self.frames_x, self.frames_y))
            self.albedo = np.mean(stack, axis=2)

        alb = (self.albedo - np.min(self.albedo)) / (np.max(self.albedo)-np.min(self.albedo))
        alb *= 255.0
        # Save the albedo as PNG
        # RBG
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            alb = cv2.cvtColor(np.array(alb, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite('Results/albedo.PNG', alb)
        # Grayscale
        else:
            cv2.imwrite('Results/albedo.PNG', np.array(alb, dtype=np.uint8))

    def computeNormalMapLookUp(self):
        # Ineffective
        if self.n > 2:
            print("Gradient shifting requires, two gradients per viewing direction. Average your data to fit.")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        """
        # Get overall maximum value
        max_norm = max(np.max(self.frames_x), np.max(self.frames_y))
        # It is required to remove the mean from the captured images
        self.frames_x = (self.frames_x / max_norm)
        self.frames_y = (self.frames_y / max_norm)
        """
        # Instead of normalizing frame by frame we normalize by the last captured image
        # this image is taken under constant maximum illumination and can therefore correct
        # for different colors and normalize pixel by pixel
        # RGB to grayscale
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            frames_x_n = np.dstack(
                (color.rgb2gray(self.frames_x[..., 0]), color.rgb2gray(self.frames_x[..., 1]))).astype(np.float64)
            frames_y_n = np.dstack(
                (color.rgb2gray(self.frames_y[..., 0]), color.rgb2gray(self.frames_y[..., 1]))).astype(np.float64)
            frame_r_n = color.rgb2gray(self.frame_reflectivity).astype(np.float64) + 1e-9
        # Grayscale
        else:
            frames_x_n = self.frames_x.astype(np.float64)
            frames_y_n = self.frames_y.astype(np.float64)
            frame_r_n = self.frame_reflectivity.astype(np.float64) + 1e-9

        for i in range(self.n):
            frames_x_n[..., i] = frames_x_n[..., i] / frame_r_n
            frames_y_n[..., i] = frames_y_n[..., i] / frame_r_n

        """
        lookup = np.load('CalibrationNumpyData/lookup.npy')
        l_x_0 = np.abs(lookup - np.repeat(np.expand_dims(frames_x_n[..., 0], axis=2), lookup.shape[2], axis=2))
        l_x_1 = np.abs(lookup - np.repeat(np.expand_dims(frames_x_n[..., 1], axis=2), lookup.shape[2], axis=2))
        l_y_0 = np.abs(lookup - np.repeat(np.expand_dims(frames_y_n[..., 0], axis=2), lookup.shape[2], axis=2))
        l_y_1 = np.abs(lookup - np.repeat(np.expand_dims(frames_y_n[..., 1], axis=2), lookup.shape[2], axis=2))
        l_x_0 = np.argmin(l_x_0, axis=2)
        l_x_1 = np.argmin(l_x_1, axis=2)
        l_y_0 = np.argmin(l_y_0, axis=2)
        l_y_1 = np.argmin(l_y_1, axis=2)
        frames_x_n[..., 0] = l_x_0
        frames_x_n[..., 1] = l_x_1
        frames_y_n[..., 0] = l_y_0
        frames_y_n[..., 1] = l_y_1
        np.save('CapturedNumpyData/capture_lookupx0.npy', l_x_0)
        np.save('CapturedNumpyData/capture_lookupx1.npy', l_x_1)
        np.save('CapturedNumpyData/capture_lookupy0.npy', l_y_0)
        np.save('CapturedNumpyData/capture_lookupy1.npy', l_y_1)
        """
        # Compute difference of gradient illuminations in opposing directions
        self.diff_x = frames_x_n[..., 1] - frames_x_n[..., 0]
        self.diff_y = frames_y_n[..., 1] - frames_y_n[..., 0]
        # Compute normals
        z = np.sqrt(1 - np.square(self.diff_x) - np.square(self.diff_y))
        norm = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y) + np.square(z))
        x = self.diff_x / norm
        y = self.diff_y / norm
        z /= norm
        # Create normals array
        self.normals = np.stack((x, y, z), axis=2)
        # Save as PNG file
        cv2.imwrite('Results/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                        cv2.COLOR_RGB2BGR))

    def computeNormalMapSingle(self, gamma):
        if self.n > 2:
            print("Gradient shifting requires, two gradients per viewing direction. Average your data to fit.")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        # Instead of normalizing frame by frame we normalize by the last captured image
        # this image is taken under constant maximum illumination and can therefore correct
        # for different colors and normalize pixel by pixel
        # RGB to grayscale

        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            frames_x_n = np.dstack(
                (color.rgb2gray(self.frames_x[..., 0]), color.rgb2gray(self.frames_x[..., 1]))).astype(np.float64)
            frames_y_n = np.dstack(
                (color.rgb2gray(self.frames_y[..., 0]), color.rgb2gray(self.frames_y[..., 1]))).astype(np.float64)
            frame_r_n = color.rgb2gray(self.frame_reflectivity).astype(np.float64) + 1e-9
        # Grayscale
        else:
            frames_x_n = self.frames_x.astype(np.float64)**gamma
            frames_y_n = self.frames_y.astype(np.float64)**gamma
            frame_r_n = self.frame_reflectivity.astype(np.float64)**gamma

        # Get overall maximum value
        max_norm = max(np.max(frames_x_n), np.max(frames_y_n))
        # It is required to remove the mean from the captured images
        frames_x_n = (frames_x_n / max_norm)
        frames_y_n = (frames_y_n / max_norm)

        # Compute difference of gradient illuminations in opposing directions
        self.diff_x = frames_x_n[..., 1] - frames_x_n[..., 0]
        self.diff_y = frames_y_n[..., 1] - frames_y_n[..., 0]
        # Compute normals
        z = np.sqrt(1 - np.square(self.diff_x) - np.square(self.diff_y))
        norm = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y) + np.square(z))
        x = self.diff_x / norm
        y = self.diff_y / norm
        z /= norm
        # Create normals array
        self.normals = np.stack((x, y, z), axis=2)
        # Save as PNG file
        cv2.imwrite('Results/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                        cv2.COLOR_RGB2BGR))
        np.save('Results/normals.npy', self.normals)

    def computeNormalMapRadiance(self, gamma):
        if self.n > 2:
            print("Gradient shifting requires, two gradients per viewing direction. Average your data to fit.")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        # Instead of normalizing frame by frame we normalize by the last captured image
        # this image is taken under constant maximum illumination and can therefore correct
        # for different colors and normalize pixel by pixel
        # RGB to grayscale
        if len(self.frames_x.shape) == 4 or len(self.frames_y.shape) == 4:
            frames_x_n = np.dstack(
                (color.rgb2gray(self.frames_x[..., 0]), color.rgb2gray(self.frames_x[..., 1]))).astype(np.float64)
            frames_y_n = np.dstack(
                (color.rgb2gray(self.frames_y[..., 0]), color.rgb2gray(self.frames_y[..., 1]))).astype(np.float64)
            frame_r_n = color.rgb2gray(self.frame_reflectivity).astype(np.float64) + 1e-9
        # Grayscale
        else:
            frames_x_n = self.frames_x.astype(np.float64)**gamma
            frames_y_n = self.frames_y.astype(np.float64)**gamma
            frame_r_n = self.frame_reflectivity.astype(np.float64)**gamma + 1e-9

        for i in range(self.n):
            frames_x_n[..., i] = frames_x_n[..., i] #/ frame_r_n
            frames_y_n[..., i] = frames_y_n[..., i] #/ frame_r_n

        # Get overall maximum value
        max_norm = max(np.max(frames_x_n), np.max(frames_y_n))
        # It is required to remove the mean from the captured images
        frames_x_n = (frames_x_n / max_norm)
        frames_y_n = (frames_y_n / max_norm)

        # Compute difference of gradient illuminations in opposing directions
        self.diff_x = frames_x_n[..., 1] - frames_x_n[..., 0]
        self.diff_y = frames_y_n[..., 1] - frames_y_n[..., 0]
        # Compute normals
        z = np.sqrt(1 - np.square(self.diff_x) - np.square(self.diff_y))
        norm = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y) + np.square(z))
        x = self.diff_x / norm
        y = self.diff_y / norm
        z /= norm
        # Create normals array
        self.normals = np.stack((x, y, z), axis=2)
        # Save as PNG file
        cv2.imwrite('Results/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                        cv2.COLOR_RGB2BGR))
        np.save('Results/normals.npy', self.normals)

    def computePointCloud(self, crop=((0, 0), (0, 0))):
        # Crop by ((start_of_crop_x, height_of_crop_x), (start_of_crop_y, height_of_crop_y))
        # IMPORTANT: Note the second tuple parameters are the cropping height/width, not the x/y crop end positions !!
        if crop[0][1] == 0 or crop[0][1] == 0:
            height = self.frames_x.shape[0]
            width = self.frames_x.shape[1]
        else:
            height = crop[0][1]
            width = crop[1][1]
        crop_x0 = crop[0][0]
        crop_y0 = crop[1][0]
        crop_x1 = crop_x0 + height
        crop_y1 = crop_y0 + width
        # Create mesh object and set normals, albedo, and compute depth
        meshGI = Mesh.Mesh("GradientIllumination", width, height)
        meshGI.setNormal(self.normals[crop_y0:crop_y1, crop_x0:crop_x1, ...])
        meshGI.setTexture(self.albedo[crop_y0:crop_y1, crop_x0:crop_x1, ...])
        self.depth = meshGI.setDepth()
        depth_png = (self.depth - np.min(self.depth)) / (np.max(self.depth) - np.min(self.depth))
        depth_png *= 255.0
        cv2.imwrite('Results/depth.PNG', depth_png)
        # Create mesh obj to export
        meshGI.exportOBJ("Results/", True)

    @staticmethod
    def normalizeRGBNormalsPQ(normals_red, normals_green, normals_blue):
        # Normals under red light
        p_red = -normals_red[..., 0] / normals_red[..., 2]
        q_red = -normals_red[..., 1] / normals_red[..., 2]
        p_red = np.nan_to_num(p_red)
        q_red = np.nan_to_num(q_red)
        # Normals under green light
        p_green = -normals_green[..., 0] / normals_green[..., 2]
        q_green = -normals_green[..., 1] / normals_green[..., 2]
        p_green = np.nan_to_num(p_green)
        q_green = np.nan_to_num(q_green)
        # Normals under blue light
        p_blue = -normals_blue[..., 0] / normals_blue[..., 2]
        q_blue = -normals_blue[..., 1] / normals_blue[..., 2]
        p_blue = np.nan_to_num(p_blue)
        q_blue = np.nan_to_num(q_blue)
        return p_red, q_red, p_green, q_green, p_blue, q_blue

    @staticmethod
    def normalizeRGNormalsPQ(normals_red, normals_green):
        # Normals under red light
        p_red = -normals_red[..., 0] / normals_red[..., 2]
        q_red = -normals_red[..., 1] / normals_red[..., 2]
        p_red = np.nan_to_num(p_red)
        q_red = np.nan_to_num(q_red)
        # Normals under green light
        p_green = -normals_green[..., 0] / normals_green[..., 2]
        q_green = -normals_green[..., 1] / normals_green[..., 2]
        p_green = np.nan_to_num(p_green)
        q_green = np.nan_to_num(q_green)
        return p_red, q_red, p_green, q_green

    @staticmethod
    def computeSegmentationRGBDepth(normals_red, normals_green, normals_blue):
        p_red, q_red, p_green, q_green, p_blue, q_blue = GradientShiftingReconstruction.normalizeRGBNormalsPQ(
            normals_red, normals_green, normals_blue)
        t1 = (p_red+(p_green+p_blue)/2)/2
        t2 = (q_red + (q_green + q_blue) / 2) / 2
        z = wavepy.surface_from_grad.frankotchellappa(t1, t2, False)
        z_all = z.real
        zr = wavepy.surface_from_grad.frankotchellappa(p_red, q_red, False)
        z_red = zr.real
        z_real = np.minimum(z_all, z_red)
        return z_real, t1, t2

    @staticmethod
    def computeGradientSpreadRGBDepth(normals_red, normals_green):
        # Normalize by z component to obtain P and Q
        p_red, q_red, p_green, q_green = GradientShiftingReconstruction.normalizeRGNormalsPQ(
            normals_red, normals_green)
        print(p_red.shape, p_green.shape, q_red.shape, q_green.shape)
        # Fourier Transform
        p_red_fft = np.fft.fft(p_red)
        p_green_fft = np.fft.fft(p_red)
        q_red_fft = np.fft.fft(q_red)
        q_green_fft = np.fft.fft(q_green)
        """
        print(p_red_fft.shape, p_green_fft.shape, q_red_fft.shape, q_green_fft.shape)
        # Meshgrid for spatial coordinates
        x = np.linspace(1, p_red.shape[1]+1, p_red.shape[1])
        y = np.linspace(1, p_red.shape[0]+1, p_red.shape[0])
        u, v = np.meshgrid(x, y)
        print(u.shape, v.shape)
        # Enforce criterium after Dong et al.
        p_r_cost = v*p_red_fft - u*q_red_fft
        p_g_cost = v*p_green_fft - u*q_green_fft
        p_cost = np.dstack((p_r_cost, p_g_cost))
        # Find gradient that minimizes constraint
        ids = np.argmin(p_cost, axis=2)
        n_ids = np.where((ids == 0) | (ids == 1), ids ^ 1, ids)
        opt_fft_p = n_ids * p_red + ids * p_green
        opt_fft_q = n_ids * q_red + ids * q_green
        """
        """
        p_stack = np.dstack((p_red_fft, q_red_fft))
        q_stack = np.dstack((q_red_fft, q_red_fft))
        print(p_stack.shape, q_stack.shape)
        # Take fourier transform of gradient that minimizes above mentioned
        opt_fft_p = p_stack[..., ids]
        opt_fft_q = q_stack[..., ids]
        
        print(opt_fft_p.shape, opt_fft_q.shape)
        """
        # Inverse Fourier Transform to acquire spatial gradients
        #opt_p = np.fft.ifft(opt_fft_p)
        #opt_q = np.fft.ifft(opt_fft_q)
        opt_p = np.fft.ifft(p_red_fft)
        opt_q = np.fft.ifft(q_red_fft)
        # Fanko Chelappa to integrate to depth values
        #opt_z = sfg.frankotchellappa(opt_p, opt_q, False)
        return opt_p.real, opt_q.real#, opt_z

    def highPassFilter(self):
        # Smooth filter operation
        kernel = np.ones((5, 5), np.float32) / 25
        self.normals = cv2.filter2D(self.normals, -1, kernel)
