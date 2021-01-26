from abc import ABC
from ImageProcessing import ImageProcessing
from Reconstructions import Mesh
import numpy as np
from skimage import color
import cv2
import torch
from tifffile import imsave

np.seterr(divide='ignore', invalid='ignore')


class DeflectometryReconstruction(ImageProcessing, ABC):

    def __init__(self, capture_path='CapturedNumpyData/capture_%i.npy', nph=4):
        super().__init__(capture_path)
        self.nph = nph
        self.frames_x = None
        self.frames_y = None
        self.phase_x = None
        self.phase_y = None
        self.magH = None
        self.magV = None
        self.normals = None
        self.depth = None
        self.albedo = None

    def loadData(self):
        #  Load first frame for dimensions and expand along 2nd axis to create a 3D array for all phase shifts
        self.frames_x = np.load(self.path % 0)
        self.frames_y = np.load(self.path % self.nph)
        convert = False
        if len(self.frames_x.shape) == 3 or len(self.frames_y.shape) == 3:
            self.frames_x = color.rgb2gray(self.frames_x)
            self.frames_y = color.rgb2gray(self.frames_y)
            convert = True
        self.frames_x = np.expand_dims(self.frames_x, axis=2)
        self.frames_y = np.expand_dims(self.frames_y, axis=2)
        # Fill
        for i in range(1, self.nph):
            if convert:
                self.frames_x = np.dstack((self.frames_x, color.rgb2gray(np.load(self.path % i))))
                self.frames_y = np.dstack((self.frames_y, color.rgb2gray(np.load(self.path % (i + self.nph)))))
            else:
                self.frames_x = np.dstack((self.frames_x, np.load(self.path % i)))
                self.frames_y = np.dstack((self.frames_y, np.load(self.path % (i + self.nph))))

    def saveTiff(self):
        # Saves phase, normals and captured images in tiff file format
        imsave('phaseX.tif', self.phase_x.astype(np.float16))
        imsave('phaseY.tif', self.phase_y.astype(np.float16))
        cv2.imwrite('normals.tif', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        imsave('unfiltered_normals.tif', self.normals.astype(np.float16))
        for i in range(self.nph):
            imsave('captureX_%i.tif' % i, self.frames_x[:, :, i].astype(np.float16))
            imsave('captureY_%i.tif' % i, self.frames_y[:, :, i].astype(np.float16))

    def computePhaseMaps(self):
        if self.nph > 4:
            print("Fit your data to a sinusoid first")
        if self.frames_y is None or self.frames_x is None:
            print("Load data first, using loadData")
        # Normalize data
        self.frames_x = (self.frames_x / np.max(self.frames_x))
        self.frames_y = (self.frames_y / np.max(self.frames_y))
        # Compute phase for x direction
        ratio_x1 = self.frames_x[:, :, 1] - self.frames_x[:, :, 3]
        ratio_x2 = self.frames_x[:, :, 0] - self.frames_x[:, :, 2]
        # self.phase_x = np.arctan(np.divide(ratio_x1, ratio_x2))
        self.phase_x = np.arctan2(ratio_x1, ratio_x2)
        # Compute phase for y direction
        ratio_y1 = self.frames_y[:, :, 1] - self.frames_y[:, :, 3]
        ratio_y2 = self.frames_y[:, :, 0] - self.frames_y[:, :, 2]
        # self.phase_y = np.arctan(np.divide(ratio_y1, ratio_y2))
        self.phase_y = np.arctan2(ratio_y1, ratio_y2)
        np.save('CapturedNumpyData/phase_x', self.phase_x)
        np.save('CapturedNumpyData/phase_y', self.phase_y)

    def computeNormalMap(self):
        if self.phase_x is None or self.phase_y is None:
            print("Compute phase maps first")
        # Create empty result matrix for surface normals
        self.normals = np.zeros((self.frames_x.shape[0], self.frames_x.shape[1], 3))
        # Compute surface normals for every pixel
        self.normals[:, :, 0] = np.multiply(
            np.divide(1, np.sqrt(np.square(self.phase_x) + np.square(self.phase_y) + 1)), self.phase_x)
        self.normals[:, :, 1] = np.multiply(
            np.divide(1, np.sqrt(np.square(self.phase_x) + np.square(self.phase_y) + 1)), self.phase_y)
        self.normals[:, :, 2] = np.divide(1, np.sqrt(np.square(self.phase_x) + np.square(self.phase_y) + 1)) * -1
        # Save as PNG file
        cv2.imwrite('CapturedImages/normals.PNG', cv2.cvtColor(np.array((self.normals + 1) / 2.0 * 255, dtype=np.uint8),
                                                               cv2.COLOR_RGB2BGR))

    def computeAlbedo(self):
        # Average over all frames to retrieve an albedo estimation
        stack = np.dstack((self.frames_x, self.frames_y))
        self.albedo = np.mean(stack, axis=2)
        # Save the albedo as PNG
        cv2.imwrite('CapturedImages/albedo.PNG', np.array(self.albedo * 255, dtype=np.uint8))

    def computePointCloud(self):
        # Retrieve resolution from captured frames
        height = self.frames_x.shape[0]
        width = self.frames_x.shape[1]
        # Create mesh object and set normals, albedo, and compute depth
        meshGI = Mesh.Mesh("Deflectometry", height, width)
        meshGI.setNormal(self.normals)
        meshGI.setTexture(self.albedo)
        self.depth = meshGI.setDepth()
        # Create mesh obj to export
        meshGI.exportOBJ("Results/", True)

    def highPassFilter(self):
        # Smooth filter operation
        kernel = np.ones((5, 5), np.float32) / 25
        self.normals = cv2.filter2D(self.normals, -1, kernel)

    def sinusoidalFitting(self):
        # Create a time interval over all frames
        t = np.linspace(0, 2 * np.pi, self.nph + 1)[:self.nph]
        # Create cosine, sine and constant arrays
        X = np.stack((np.sin(t), np.cos(t), np.ones(t.shape)), 1)
        # Retrieve resolution from frames
        W = self.frames_x.shape[0]
        H = self.frames_x.shape[1]
        Xt = torch.tensor(X).expand(H * W, self.nph, 3)
        framesR_y = self.frames_y.swapaxes(0, 2).swapaxes(1, 2)*1.0
        framesR_x = self.frames_x.swapaxes(0, 2).swapaxes(1, 2)*1.0
        imsHt = torch.tensor(framesR_x).permute(1, 2, 0).view(H * W, self.nph)
        imsVt = torch.tensor(framesR_y).permute(1, 2, 0).view(H * W, self.nph)
        # Compute sinusoidal fit for phase in y direction
        A = torch.matmul(torch.matmul(torch.matmul(Xt.transpose(1, 2), Xt).inverse(),
                                      Xt.transpose(1, 2)), imsHt.unsqueeze(2)).squeeze()
        # Save magnitude in horizontal (y) direction
        self.magH = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2).view(H, W)
        phase_y = torch.atan2(A[:, 0], A[:, 1]).view(W, H)
        # Save corrected phase y
        self.phase_y = phase_y.numpy()
        bgH = A[:, 2].view(H, W)
        # Compute sinusoidal fit for phase in y direction
        AV = torch.matmul(torch.matmul(torch.matmul(Xt.transpose(1, 2), Xt).inverse(),
                                       Xt.transpose(1, 2)), imsVt.unsqueeze(2)).squeeze()
        # Save magnitude in vertical (x) direction
        self.magV = torch.sqrt(AV[:, 0] ** 2 + AV[:, 1] ** 2).view(H, W)
        phase_x = torch.atan2(AV[:, 0], AV[:, 1]).view(W, H)
        # Save corrected phase y
        self.phase_x = phase_x.numpy()


        

