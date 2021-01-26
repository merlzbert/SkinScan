from __future__ import print_function
import os, sys, glob
import numpy as np
import cv2
import scipy.spatial
import matplotlib
import wavepy
import matplotlib.pyplot as plt
# import pyvista as pv
# import plotly.graph_objects as go

# Set Parameters
DEFLECTOMETRY_GRADMAP_FILTERSIZE = 49

# Render parameters
MESH_Ka = [0.0, 0.0, 0.0]
MESH_Kd = [0.9, 0.9, 0.9]
MESH_Ks = [0.1, 0.1, 0.1]
MESH_Ns = 5.000000
MESH_d = 0.010000
MESH_illum = 1

# Scaling factors
MESH_DEPTH_SCALINGFACTOR = 5
MESH_SIGMA_GLOBCORR = 5


class Mesh:
    def __init__(self, name, height, width, cropMask=0):

        self.name = name
        self.height = height
        self.width = width
        self.vertex = np.zeros((self.height * self.width, 3))
        # set 2d plane
        self.setVertex2D()
        self.setFace()
        if cropMask == 0:
            self.mask = np.array([0, 0, self.height, self.width])
        else:
            self.mask = cropMask

    def setVertex2D(self):
        [i, j] = np.meshgrid(np.r_[1:self.height + 1], np.r_[1:self.width + 1])
        self.vertex2D = np.concatenate((i.flatten('F')[:, None], j.flatten('F')[:, None]), axis=1).astype(np.int16)
        self.vertex[:, 0:2] = self.vertex2D

    # options = 'Qt QbB Qc'
    # self.faces = scipy.spatial.Delaunay(v, qhull_options=options).simplices + 1

    def setFace(self):
        # N = 2
        options = 'Qt QbB Qc'
        self.faces = scipy.spatial.Delaunay(self.vertex2D, qhull_options=options).simplices + 1

    def setDepth(self):
        # Set x and y direction
        p = self.normals[..., 0] / self.normals[..., 2]
        q = self.normals[..., 1] / self.normals[..., 2]
        p = np.nan_to_num(p)
        q = np.nan_to_num(q)
        # TODO: Low pass filter to filter out the noise (BilateralFilter)
        # Compute depth as a imaginary number (real: depth, img: noise)
        # Using Frankot Chellappa method with FFT
        #x = sfg.frankotchellappa(p, q, False)
        x = wavepy.surface_from_grad.frankotchellappa(p, q, False)
        depth = x.real
        # depth_smooth = cv2.GaussianBlur(depth, (sigma_GlobCorr, sigma_GlobCorr), ((sigma_GlobCorr + 1) / 2))
        depth *= MESH_DEPTH_SCALINGFACTOR
        # cv.bilateralFilter(src, dst, 9, 75, 75, cv.BORDER_DEFAULT)
        depth_corr = cv2.bilateralFilter(depth.astype(np.float32), MESH_SIGMA_GLOBCORR, 90, 90)
        # depth_corr = depth
        # cv2.imwrite('./depth_before.png', (depth*255).astype(np.uint8))
        # depth = cv2.blur(depth.astype(np.float32), (5,5))
        # cv2.imwrite('./depth_after.png', (depth*255).astype(np.uint8))
        depth_crop = depth_corr[self.mask[0]:self.mask[2], self.mask[1]:self.mask[3]]
        self.vertex[:, 2] = depth_crop.reshape(-1)
        return depth

    def setNormal(self, normals):
        self.normals = normals

    def setTexture(self, texture):
        #self.texture = (cv2.flip(texture, -1) * 255).astype(np.uint8)
        self.texture = cv2.flip(texture, -1)


    # @nb.jit(nopython=True)
    def exportOBJ(self, path, withTexture=True):
        # totalCount = self.vertex.shape[0] + self.height * self.width + self.faces.shape[0]
        # progress = ProgressBar(totalCount, fmt=ProgressBar.FULL)
        print('Exporting OBJ...')
        N = self.normals[self.mask[0]:self.mask[2], self.mask[1]:self.mask[3]].reshape(-1, 3)
        filename = os.path.join(os.path.normpath(path), "mesh_" + self.name + ".obj")
        f = open(filename, 'w')
        f.write('mtllib material_' + self.name + '.mtl\n')
        f.write('usemtl Textured\n')
        for i in range(self.vertex.shape[0]):
            # progress.current += 1
            # progress()
            f.write('v %f %f %f\n' % (self.vertex[i, 0], self.vertex[i, 1], self.vertex[i, 2]))
            f.write('vn %f %f %f\n' % (N[i, 0], N[i, 1], N[i, 2]))

        if withTexture:
            for i in range(self.height):
                for j in range(self.width):
                    # progress.current += 1
                    # progress()
                    u = (i) / self.height
                    v = 1 - (j) / self.width
                    f.write('vt %f %f\n' % (v, u))

        for i in range(self.faces.shape[0]):
            # progress.current += 1
            # progress()
            # f.write('f ' + str(self.faces[i, 0]) + ' ' + str(self.faces[i, 1]) + ' ' + str(self.faces[i, 2]) + '\n')
            f.write(
                'f ' + str(self.faces[i, 0]) + '/' + str(self.faces[i, 0]) + '/' + str(self.faces[i, 0]) + ' ' + str(
                    self.faces[i, 1]) + '/' + str(self.faces[i, 1]) + '/' + str(self.faces[i, 1]) + ' ' + str(
                    self.faces[i, 2]) + '/' + str(self.faces[i, 2]) + '/' + str(self.faces[i, 2]) + '\n')

        f.close()

        if withTexture:
            # export mtl
            filename = os.path.join(os.path.normpath(path), "material_" + self.name + ".mtl")
            f = open(filename, 'w')
            f.write('newmtl Textured\n')
            f.write('Ka %f %f %f\n' % (MESH_Ka[0], MESH_Ka[1], MESH_Ka[2]))
            f.write('Kd %f %f %f\n' % (MESH_Kd[0], MESH_Kd[1], MESH_Kd[2]))
            f.write('Ks %f %f %f\n' % (MESH_Ks[0], MESH_Ks[1], MESH_Ks[2]))
            f.write('Ns %f\n' % (MESH_Ns))
            f.write('d %f\n' % (MESH_d))
            f.write('illum %d\n' % (MESH_illum))
            f.write("map_Kd texture_" + self.name + ".jpg\n")
            f.close()

            # export texture
            filename = os.path.join(os.path.normpath(path), "texture_" + self.name + ".jpg")
            # cv2.imwrite(filename, self.texture)
            # cv2.imwrite(filename, np.array((utilities.imgBrighten(self.texture, 10) * 255), dtype=np.uint8))
            cv2.imwrite(filename, self.texture)

